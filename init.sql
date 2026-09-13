-- ==============================================================================
-- SentinelRAG: Relational Schema & Indices (init.sql)
-- ==============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Document registry table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id VARCHAR(255) UNIQUE NOT NULL,
    title VARCHAR(512) NOT NULL,
    classification VARCHAR(32) NOT NULL CHECK (classification IN ('public', 'internal', 'confidential', 'executive')),
    allowed_roles TEXT[] NOT NULL DEFAULT '{"*"}',
    version INT NOT NULL DEFAULT 1,
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Chunk registry & hashing table
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY, -- Deterministic UUIDv5
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content_hash CHAR(64) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_doc_chunk_idx UNIQUE (document_id, chunk_index)
);

-- Access audit log
CREATE TABLE IF NOT EXISTS retrieval_audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    user_roles TEXT[] NOT NULL,
    user_clearance VARCHAR(32) NOT NULL,
    query_text TEXT NOT NULL,
    retrieved_chunk_ids TEXT[] NOT NULL,
    execution_time_ms FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Performance and lookup indices
CREATE INDEX IF NOT EXISTS idx_docs_classification ON documents(classification);
CREATE INDEX IF NOT EXISTS idx_docs_external_id ON documents(external_id);
CREATE INDEX IF NOT EXISTS idx_docs_is_deleted ON documents(is_deleted);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON document_chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_audit_user_id ON retrieval_audit_logs(user_id);
