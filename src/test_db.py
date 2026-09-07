import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

conn = psycopg.connect(
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
)

with conn.cursor() as cur:
    cur.execute("SELECT version();")
    print(cur.fetchone()[0])

    cur.execute("""
        SELECT extversion
        FROM pg_extension
        WHERE extname = 'vector';
    """)
    print("pgvector:", cur.fetchone()[0])

conn.close()

print("Database connection successful.")