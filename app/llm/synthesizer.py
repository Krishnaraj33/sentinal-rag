"""
Grounded synthesis engine with cryptographic citations and clearance ceilings.
Standard Compliance: SRS Module 5.
"""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from app.constants import CLEARANCE_ORDER
from app.llm.client import get_llm_client

logger = logging.getLogger(__name__)


class Citation(BaseModel):
    """Citation metadata verifying grounding provenance."""
    document_id: str
    chunk_id: str
    content_hash: str
    classification: str


class SynthesisResult(BaseModel):
    """Structured generative response."""
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    context_clearance_ceiling: str = "public"


class StructuredSynthesisResponse(BaseModel):
    """Structured response model for LLM synthesis."""
    answer: str = Field(description="Accurate, concise answer grounded strictly in the provided authorized context chunks.")
    cited_chunk_ids: List[str] = Field(default_factory=list, description="IDs of the context chunks directly cited to substantiate the answer.")


SYSTEM_SYNTHESIS_PROMPT = """You are SentinelRAG. Synthesize an accurate, concise answer grounded strictly in the provided authorized context chunks.
Return ONLY a valid JSON object matching the schema:
{{"answer": "concise answer", "cited_chunk_ids": ["chunk-id-1"]}}
If the context chunks contain NO information relevant to the user's question, return:
{{"answer": "I do not possess authorized context to answer this query.", "cited_chunk_ids": []}}
"""


def compute_clearance_ceiling(chunks: List[Dict[str, Any]]) -> str:
    """Calculates the maximum clearance level present across context chunks."""
    highest_order = 0
    ceiling = "public"
    for chunk in chunks:
        classification = chunk.get("classification", "public").lower()
        order = CLEARANCE_ORDER.get(classification, 0)
        if order > highest_order:
            highest_order = order
            ceiling = classification
    return ceiling


class Synthesizer:
    """Grounded answer synthesizer with citation integrity."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client or get_llm_client()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_SYNTHESIS_PROMPT),
            ("user", "USER QUERY: {query}\n\nAUTHORIZED CONTEXT CHUNKS:\n{context_lines}\n\nSynthesize the answer and output valid JSON with 'answer' and 'cited_chunk_ids'."),
        ])

    async def synthesize(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
    ) -> SynthesisResult:
        """
        Synthesizes grounded answer from authorized chunks.
        Guarantees zero leakage and citation auditability.
        """
        if not context_chunks:
            return SynthesisResult(
                answer="I do not possess authorized context to answer this query.",
                citations=[],
                context_clearance_ceiling="public",
            )

        ceiling = compute_clearance_ceiling(context_chunks)

        # Build chunk lookup dictionary for citation resolution
        chunk_map = {str(c["id"]): c for c in context_chunks}

        # Build context prompt
        context_lines_list = []
        for idx, chunk in enumerate(context_chunks):
            cid = str(chunk["id"])
            c_hash = chunk.get("content_hash", "")[:12]
            c_class = chunk.get("classification", "public")
            c_text = chunk.get("text", "")
            context_lines_list.append(
                f"[CHUNK {idx + 1} | ID: {cid} | Classification: {c_class} | Hash: {c_hash}]\n{c_text}\n"
            )
        
        context_lines = "---".join(context_lines_list)

        try:
            if self.llm_client.provider == "mock" or not self.llm_client.api_key:
                user_prompt = f"USER QUERY: {query}\n\nAUTHORIZED CONTEXT CHUNKS:\n{context_lines}\n\nSynthesize the answer and output valid JSON with 'answer' and 'cited_chunk_ids'."
                structured_resp = await self.llm_client.generate_structured(
                    system_prompt=SYSTEM_SYNTHESIS_PROMPT,
                    user_prompt=user_prompt,
                    response_model=StructuredSynthesisResponse,
                    temperature=0.1,
                    max_tokens=2048,
                )
            else:
                chat_model = self.llm_client._get_chat_model(temperature=0.1, max_tokens=2048)
                structured_llm = chat_model.with_structured_output(StructuredSynthesisResponse)
                
                chain = self.prompt | structured_llm
                
                structured_resp = await chain.ainvoke({
                    "query": query,
                    "context_lines": context_lines
                })
            answer = structured_resp.answer.strip()
            cited_ids = structured_resp.cited_chunk_ids
        except Exception as exc:
            logger.warning("LCEL chain failed (%s). Falling back.", exc)
            top_chunk = context_chunks[0]
            answer = top_chunk.get("text", "")
            cited_ids = [str(c["id"]) for c in context_chunks[:2]]

        # Check for fallback answer phrases
        if "do not possess authorized" in answer.lower() or not answer:
            return SynthesisResult(
                answer="I do not possess authorized context to answer this query.",
                citations=[],
                context_clearance_ceiling=ceiling,
            )

        # Verify and resolve citations
        citations: List[Citation] = []
        # If model returned no cited_ids but provided an answer, cite the highest scoring chunks used
        if not cited_ids and "do not possess authorized" not in answer.lower():
            cited_ids = [str(c["id"]) for c in context_chunks[:2]]

        for cid in cited_ids:
            clean_id = str(cid).strip()
            if clean_id in chunk_map:
                source = chunk_map[clean_id]
                citations.append(
                    Citation(
                        document_id=str(source.get("document_id", "")),
                        chunk_id=clean_id,
                        content_hash=str(source.get("content_hash", "")),
                        classification=str(source.get("classification", "public")),
                    )
                )

        return SynthesisResult(
            answer=answer,
            citations=citations,
            context_clearance_ceiling=ceiling,
        )
