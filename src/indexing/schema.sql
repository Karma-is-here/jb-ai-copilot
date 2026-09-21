CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,

    document_id TEXT NOT NULL,

    text TEXT NOT NULL,

    -- Dense semantic representation
    embedding vector(384),

    -- PostgreSQL lexical-search representation
    search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', text)
    ) STORED,

    -- Vanka provenance
    source_file TEXT,
    page_start INTEGER,
    page_end INTEGER,
    section TEXT,

    chunking_strategy TEXT,
    chunk_index INTEGER,

    metadata JSONB,

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Fast document filtering
CREATE INDEX IF NOT EXISTS chunks_document_id_idx
    ON chunks (document_id);

-- HNSW index for semantic/vector retrieval
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks
    USING hnsw (embedding vector_cosine_ops);

-- GIN index for lexical/full-text retrieval
CREATE INDEX IF NOT EXISTS chunks_search_vector_idx
    ON chunks
    USING GIN (search_vector);