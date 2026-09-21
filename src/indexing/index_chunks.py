import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

CHUNKS_DIR = Path("data/output/chunks")
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32


# ---------------------------------------------------------
# Database connection
# ---------------------------------------------------------

def get_connection():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


# ---------------------------------------------------------
# Load Vanka chunks
# ---------------------------------------------------------

def load_chunks(chunks_dir: Path):
    chunks = []

    files = sorted(chunks_dir.rglob("*.json"))

    print(f"Found {len(files)} Vanka chunk files.")

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            artifact = json.load(f)

        for chunk in artifact["chunks"]:
            source = chunk.get("source", {})
            chunking = chunk.get("chunking", {})

            chunks.append({
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "text": chunk["text"],
                "source_file": source.get("file"),
                "page_start": source.get("page_start"),
                "page_end": source.get("page_end"),
                "section": source.get("section"),
                "chunking_strategy": chunking.get("strategy"),
                "chunk_index": chunking.get("chunk_index"),
                "metadata": chunk.get("metadata", {}),
            })

    print(f"Loaded {len(chunks)} chunks.")

    return chunks


# ---------------------------------------------------------
# Insert chunks + embeddings
# ---------------------------------------------------------

def index_chunks(chunks, model):
    conn = get_connection()

    try:
        with conn.cursor() as cur:

            for start in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[start:start + BATCH_SIZE]

                texts = [chunk["text"] for chunk in batch]

                embeddings = model.encode(
                    texts,
                    batch_size=BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                for chunk, embedding in zip(batch, embeddings):

                    embedding_list = embedding.tolist()

                    cur.execute(
                        """
                        INSERT INTO chunks (
                            chunk_id,
                            document_id,
                            text,
                            embedding,
                            source_file,
                            page_start,
                            page_end,
                            section,
                            chunking_strategy,
                            chunk_index,
                            metadata
                        )
                        VALUES (
                            %s, %s, %s, %s::vector,
                            %s, %s, %s, %s,
                            %s, %s, %s
                        )
                        ON CONFLICT (chunk_id)
                        DO UPDATE SET
                            document_id = EXCLUDED.document_id,
                            text = EXCLUDED.text,
                            embedding = EXCLUDED.embedding,
                            source_file = EXCLUDED.source_file,
                            page_start = EXCLUDED.page_start,
                            page_end = EXCLUDED.page_end,
                            section = EXCLUDED.section,
                            chunking_strategy = EXCLUDED.chunking_strategy,
                            chunk_index = EXCLUDED.chunk_index,
                            metadata = EXCLUDED.metadata;
                        """,
                        (
                            chunk["chunk_id"],
                            chunk["document_id"],
                            chunk["text"],
                            embedding_list,
                            chunk["source_file"],
                            chunk["page_start"],
                            chunk["page_end"],
                            chunk["section"],
                            chunk["chunking_strategy"],
                            chunk["chunk_index"],
                            json.dumps(chunk["metadata"]),
                        ),
                    )

                conn.commit()

                print(
                    f"Indexed {min(start + BATCH_SIZE, len(chunks))}"
                    f"/{len(chunks)} chunks"
                )

    finally:
        conn.close()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    if not CHUNKS_DIR.exists():
        raise FileNotFoundError(
            f"Chunk directory not found: {CHUNKS_DIR}"
        )

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    dimension = model.get_sentence_embedding_dimension()

    if dimension != 384:
        raise ValueError(
            f"Expected 384-dimensional embeddings, got {dimension}"
        )

    chunks = load_chunks(CHUNKS_DIR)

    if not chunks:
        raise RuntimeError("No chunks found.")

    index_chunks(chunks, model)

    print()
    print("Indexing complete.")


if __name__ == "__main__":
    main()