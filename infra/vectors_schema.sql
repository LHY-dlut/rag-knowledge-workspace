CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS "chunk_vector" (
    "id" VARCHAR(36) NOT NULL PRIMARY KEY,
    "owner_id" VARCHAR(36) NOT NULL,
    "kb_id" VARCHAR(36) NOT NULL,
    "doc_id" VARCHAR(36) NOT NULL,
    "parent_id" VARCHAR(36) NOT NULL,
    "content" TEXT NOT NULL,
    "metadata" JSONB NOT NULL,
    "embedding_fingerprint" VARCHAR(160) NOT NULL,
    "embedding" VECTOR(1024) NOT NULL
);
CREATE INDEX IF NOT EXISTS "idx_chunk_vecto_owner_i_d35eb2" ON "chunk_vector" ("owner_id");
CREATE INDEX IF NOT EXISTS "idx_chunk_vecto_kb_id_03a225" ON "chunk_vector" ("kb_id");
CREATE INDEX IF NOT EXISTS "idx_chunk_vecto_doc_id_cd76aa" ON "chunk_vector" ("doc_id");
CREATE INDEX IF NOT EXISTS "idx_chunk_vecto_parent__456425" ON "chunk_vector" ("parent_id");
CREATE INDEX IF NOT EXISTS chunk_vector_hnsw_idx ON chunk_vector USING hnsw (embedding vector_cosine_ops);
