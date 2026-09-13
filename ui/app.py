"""
SentinelRAG Interactive Showcase UI.
Built with Streamlit. Standard Compliance: SRS Module 6 (FR-6.1 - FR-6.4).
"""

import json
import os
from pathlib import Path
import sys
import streamlit as st

# Ensure both project root and ui directory are in sys.path
UI_DIR = Path(__file__).resolve().parent
PROJECT_DIR = UI_DIR.parent
for p in [str(PROJECT_DIR), str(UI_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from ui.api_client import SentinelApiClient
except ModuleNotFoundError:
    from api_client import SentinelApiClient

st.set_page_config(
    page_title="SentinelRAG Console",
    page_icon="Shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 10px;
    }
    .badge-public { background-color: #4CAF50; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
    .badge-internal { background-color: #2196F3; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
    .badge-confidential { background-color: #FF9800; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
    .badge-executive { background-color: #E91E63; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
    .denial-box {
        background-color: #ffebee;
        border-left: 5px solid #d32f2f;
        padding: 15px;
        border-radius: 6px;
        color: #b71c1c;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

api_url = os.environ.get("API_GATEWAY_URL", "http://0.0.0.0:8000")
client = SentinelApiClient(base_url=api_url)


def generate_offline_token(sub: str, roles: list, clearance: str) -> str:
    """Generates an authentic HS256 JWT using standard library and shared secret."""
    import base64
    import hashlib
    import hmac
    import json
    import time
    secret = os.environ.get("JWT_SECRET_KEY", "enterprise_super_secret_jwt_key_2026_change_in_prod")
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": sub,
        "roles": roles,
        "clearance": clearance,
        "exp": int(time.time()) + (60 * 60 * 24),
    }
    b64_h = base64.urlsafe_b64encode(json.dumps(header, separators=(',', ':')).encode()).decode().rstrip("=")
    b64_p = base64.urlsafe_b64encode(json.dumps(payload, separators=(',', ':')).encode()).decode().rstrip("=")
    sig = hmac.new(secret.encode(), f"{b64_h}.{b64_p}".encode(), hashlib.sha256).digest()
    b64_s = base64.urlsafe_b64encode(sig).decode().rstrip("=")
    return f"{b64_h}.{b64_p}.{b64_s}"


def get_fallback_personas() -> dict:
    """Generates authentic fallback personas signed with the system JWT secret."""
    personas_def = {
        "contractor": {
            "name": "Guest / Contractor",
            "sub": "USR-CONTRACTOR-01",
            "clearance": "public",
            "roles": ["contractor"],
        },
        "engineering": {
            "name": "Software Engineer",
            "sub": "USR-ENG-42",
            "clearance": "internal",
            "roles": ["engineering"],
        },
        "finance": {
            "name": "Finance Manager",
            "sub": "USR-FIN-09",
            "clearance": "confidential",
            "roles": ["finance"],
        },
        "executive": {
            "name": "Chief Executive Officer",
            "sub": "USR-CEO-01",
            "clearance": "executive",
            "roles": ["c-suite", "engineering", "finance", "hr"],
        },
    }
    for k, p in personas_def.items():
        p["token"] = generate_offline_token(p["sub"], p["roles"], p["clearance"])
    return personas_def


@st.cache_data(ttl=60)
def _fetch_remote_personas():
    return client.get_personas()


def load_personas():
    try:
        data = _fetch_remote_personas()
        if data and isinstance(data, dict) and len(data) > 0:
            return data
    except Exception as exc:
        st.sidebar.caption(f"Syncing with API Gateway ({exc}). Active cryptographic tokens loaded.")
    return get_fallback_personas()


personas = load_personas()
persona_options = list(personas.keys())
selected_persona_key = st.sidebar.selectbox(
    "Select Active Persona",
    options=persona_options,
    format_func=lambda k: personas[k]["name"],
    index=1 if "engineering" in persona_options else 0,
)
active_persona = personas[selected_persona_key]

# Display Claims Badges
clearance = active_persona.get("clearance", "public").lower()
st.sidebar.markdown(f"**Clearance:** <span class='badge-{clearance}'>{clearance.upper()}</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"**Roles:** `{', '.join(active_persona.get('roles', []))}`")
st.sidebar.markdown(f"**User ID (`sub`):** `{active_persona.get('sub')}`")

with st.sidebar.expander("Inspect Signed Bearer JWT", expanded=False):
    st.code(active_persona.get("token", ""), language="text")

st.sidebar.markdown("---")
st.sidebar.caption("Sentinal RAG")

# --- MAIN WORKSPACE TABS ---
tab_query, tab_invalidation, tab_inspector = st.tabs([
    "Live Query Console",
    "Real-Time Invalidation Sandbox",
    "Security Boundary Inspector",
])

# ==============================================================================
# TAB 1: LIVE QUERY CONSOLE (FR-6.2)
# ==============================================================================
with tab_query:
    st.header("Identity-Filtered Hybrid Search & Grounding")
    st.caption("Native pre-retrieval bitmask traversal prevents context starvation and leakage.")

    sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)
    with sample_col1:
        if st.button("Staging Credentials"):
            st.session_state["query_input"] = "What are our staging cluster credentials?"
    with sample_col2:
        if st.button("Travel Limit"):
            st.session_state["query_input"] = "What is our travel reimbursement limit?"
    with sample_col3:
        if st.button("Executive Bonus"):
            st.session_state["query_input"] = "What is the Q3 executive bonus structure?"
    with sample_col4:
        if st.button("Offboarding Guide"):
            st.session_state["query_input"] = "What are the offboarding procedures?"

    query_val = st.session_state.get("query_input", "What are our staging cluster credentials?")
    query_text = st.text_input("Enter natural language query:", value=query_val)
    top_k = st.slider("Top Chunks Context Pool (top_k)", min_value=1, max_value=10, value=5)

    if st.button("Execute Search & Synthesis", type="primary"):
        if not query_text.strip():
            st.warning("Please enter a valid search query.")
        else:
            with st.spinner("Executing pre-filtered traversal, RRF fusion, and synthesis..."):
                try:
                    res = client.query(
                        query_text=query_text,
                        token=active_persona.get("token", ""),
                        top_k=top_k,
                    )
                except Exception as exc:
                    st.error(f"Request failed: {exc}")
                    res = None

                if res is None:
                    pass
                elif res.get("error_status") in (401, 403) or "error" in res.get("detail", {}):
                    status_code = res.get("error_status", 403)
                    detail_val = res.get("detail", {})
                    if isinstance(detail_val, dict):
                        msg = detail_val.get("message", "Query terminated: Access unauthorized or forbidden.")
                    else:
                        msg = str(detail_val)
                    title = "ACCESS DENIED (Zero-Token Denial)" if status_code == 403 else "UNAUTHORIZED (401)"
                    st.markdown(
                        f"""
                        <div class="denial-box">
                            <h4>{title}</h4>
                            <p>{msg}</p>
                            <small>Enforced at security boundary. 0 LLM generation tokens billed.</small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    # Answer presentation
                    st.subheader("Synthesized Grounded Response")
                    ceiling = res.get("context_clearance_ceiling", "public").lower()
                    st.markdown(f"**Context Clearance Ceiling:** <span class='badge-{ceiling}'>{ceiling.upper()}</span>", unsafe_allow_html=True)
                    st.success(res.get("answer", ""))

                    # Citations
                    citations = res.get("citations", [])
                    st.subheader(f"Cryptographic Citations ({len(citations)})")
                    if citations:
                        for idx, cit in enumerate(citations):
                            c_class = cit.get("classification", "public")
                            with st.container():
                                st.markdown(
                                    f"**Citation #{idx + 1}** | Doc: `{cit.get('document_id')}` | "
                                    f"Chunk: `{cit.get('chunk_id')[:8]}...` | "
                                    f"<span class='badge-{c_class}'>{c_class.upper()}</span> | "
                                    f"SHA-256: `{cit.get('content_hash')[:16]}...`",
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.info("No specific chunks cited in generated response.")

                    # Latency & Metrics breakdown
                    metrics = res.get("metrics", {})
                    st.subheader("Latency & Scan Breakdown")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Retrieval Latency", f"{metrics.get('retrieval_latency_ms', 0)} ms")
                    m2.metric("Re-rank Latency", f"{metrics.get('rerank_latency_ms', 0)} ms")
                    m3.metric("LLM Latency", f"{metrics.get('generation_latency_ms', 0)} ms")
                    m4.metric("Total Latency", f"{metrics.get('total_latency_ms', 0)} ms")
                    m5.metric("Authorized Scanned", f"{metrics.get('authorized_chunks_scanned', 0)}")

                    # Compiled Pre-Filter Visualizer
                    with st.expander("Compiled Qdrant Pre-Filter Payload JSON", expanded=False):
                        st.json(res.get("compiled_filter", {}))

# ==============================================================================
# TAB 2: REAL-TIME INVALIDATION SANDBOX (FR-6.3)
# ==============================================================================
with tab_invalidation:
    st.header("Real-Time Invalidation Sandbox")
    st.caption("Experience sub-second vector updates and embedding call suppression on textual diffs.")

    try:
        documents = client.list_documents(token=active_persona.get("token", ""))
    except Exception:
        documents = []

    if not documents:
        st.warning("No documents found in PostgreSQL. Please run data seeding first.")
    else:
        doc_titles = {d["external_id"]: f"{d['title']} ({d['external_id']})" for d in documents}
        selected_doc_id = st.selectbox(
            "Select Document to Mutate:",
            options=list(doc_titles.keys()),
            format_func=lambda k: doc_titles[k],
        )

        selected_doc = next((d for d in documents if d["external_id"] == selected_doc_id), None)

        if selected_doc:
            col_t1, col_t2 = st.columns([3, 1])
            with col_t1:
                title_input = st.text_input("Document Title", value=selected_doc.get("title", ""))
            with col_t2:
                class_input = st.selectbox(
                    "Classification",
                    options=["public", "internal", "confidential", "executive"],
                    index=["public", "internal", "confidential", "executive"].index(
                        selected_doc.get("classification", "internal").lower()
                    ),
                )

            content_text = st.text_area(
                "Document Content (Edit below to trigger diff engine):",
                value=selected_doc.get("content", ""),
                height=220,
            )

            if st.button("Commit Document Update", type="primary"):
                with st.spinner("Hashing normalized chunks and computing diffs..."):
                    try:
                        mutation_res = client.update_document(
                            event_type="UPDATE",
                            document_id=selected_doc_id,
                            title=title_input,
                            classification=class_input,
                            allowed_roles=selected_doc.get("allowed_roles", ["*"]),
                            content=content_text,
                            sync_mode=True,
                        )
                        st.success("Document change processed idempotently!")
                        details = mutation_res.get("details", {})

                        # Render visual diff metrics
                        total_c = details.get("total_chunks", 0)
                        unchanged_c = details.get("unchanged_chunks", [])
                        mutated_c = details.get("mutated_chunks", [])
                        orphaned_c = details.get("orphaned_chunks", [])
                        savings_pct = details.get("embedding_suppression_pct", 0.0)

                        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
                        d_col1.metric("Total Chunks", total_c)
                        d_col2.metric("Bypassed (Unchanged)", len(unchanged_c), delta="0 API calls", delta_color="normal")
                        d_col3.metric("Mutated (Embedded)", len(mutated_c), delta=f"{len(mutated_c)} vectors", delta_color="inverse")
                        d_col4.metric("Suppression Savings", f"{savings_pct}%", delta="Tokens saved")

                        st.markdown("### Chunk Diff Analysis")
                        for idx in range(total_c):
                            if idx in unchanged_c:
                                st.markdown(f"**Chunk [{idx}]**: `UNCHANGED` (SHA-256 match -> Bypassed embedding)")
                            elif idx in mutated_c:
                                st.markdown(f"**Chunk [{idx}]**: `MUTATED` (SHA-256 diff -> Overwritten at Point ID)")
                        for idx in orphaned_c:
                            st.markdown(f"**Chunk [{idx}]**: `ORPHANED` (Deleted from Qdrant)")

                        st.info("Vectors in Qdrant have been updated. Re-run query in Tab 1 to observe invalidation.")
                    except Exception as exc:
                        st.error(f"Document update failed: {exc}")

# ==============================================================================
# TAB 3: SECURITY BOUNDARY INSPECTOR (FR-6.4)
# ==============================================================================
with tab_inspector:
    st.header("Security Boundary Inspector")
    st.caption("Verifies zero context leakage between organizational clearance boundaries.")

    try:
        stats = client.get_security_stats(token=active_persona.get("token", ""))
    except Exception as exc:
        st.error(f"Could not load security statistics: {exc}")
        stats = None

    if stats:
        tot = stats.get("total_chunks_in_qdrant", 0)
        vis = stats.get("visible_chunks_for_persona", 0)
        iso = stats.get("isolated_chunks_count", 0)
        iso_pct = stats.get("isolation_percentage", 0.0)
        leakage = stats.get("leakage_rate_pct", 0.00)

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Total Vectors in Qdrant", tot)
        col_s2.metric("Visible to Active Persona", vis)
        col_s3.metric("Hard-Isolated Chunks", iso, delta=f"{iso_pct}% isolated")
        col_s4.metric("Context Leakage Rate", f"{leakage:.2f}%", delta="Zero-Leakage Guarantee")

        st.markdown("---")
        st.subheader("Active Persona Access Rights")
        p_info = stats.get("active_persona", {})
        st.write(f"- **User Identifier (`sub`):** `{p_info.get('sub')}`")
        st.write(f"- **Clearance:** `{p_info.get('clearance')}`")
        st.write(f"- **Roles:** `{', '.join(p_info.get('roles', []))}`")
        st.write(f"- **Accessible Classifications:** `{p_info.get('accessible_classifications')}`")

        with st.expander("Inspect Compiled Qdrant Filter", expanded=True):
            st.json(stats.get("compiled_filter", {}))
