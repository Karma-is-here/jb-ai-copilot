import os

import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"


class VectorRetriever:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def _get_connection(self):
        return psycopg.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
        )

    def search(self, query: str, top_k: int = 10):
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        embedding = query_embedding.tolist()

        conn = self._get_connection()

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        chunk_id,
                        document_id,
                        text,
                        source_file,
                        page_start,
                        page_end,
                        section,
                        chunking_strategy,
                        chunk_index,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM chunks
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (
                        embedding,
                        embedding,
                        top_k,
                    ),
                )

                rows = cur.fetchall()

                return [
                    {
                        "chunk_id": row[0],
                        "document_id": row[1],
                        "text": row[2],
                        "source_file": row[3],
                        "page_start": row[4],
                        "page_end": row[5],
                        "section": row[6],
                        "chunking_strategy": row[7],
                        "chunk_index": row[8],
                        "similarity": float(row[9]),
                    }
                    for row in rows
                ]

        finally:
            conn.close()