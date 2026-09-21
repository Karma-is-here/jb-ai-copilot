import os

import psycopg
from dotenv import load_dotenv


load_dotenv()


class LexicalRetriever:

    def _get_connection(self):
        return psycopg.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
        )

    def search(self, query: str, top_k: int = 10):

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

                        ts_rank_cd(
                            search_vector,
                            websearch_to_tsquery('english', %s)
                        ) AS lexical_score

                    FROM chunks

                    WHERE search_vector @@
                          websearch_to_tsquery('english', %s)

                    ORDER BY lexical_score DESC

                    LIMIT %s;
                    """,
                    (
                        query,
                        query,
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
                        "lexical_score": float(row[9]),
                    }
                    for row in rows
                ]

        finally:
            conn.close()