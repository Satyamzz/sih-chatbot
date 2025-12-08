import os
import time
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# 1. FIXED: likely typo in env var name
api_key = os.getenv("PINECONE_API_KEY") 
if not api_key:
    print("Error: API Key not found. Check your .env file.")
    exit()

pc = Pinecone(api_key=api_key)
index_name = "mongo-sync-index"
index=pc.Index(index_name)
# 2. Check if index exists (optional, but safer)
existing_indexes = [i['name'] for i in pc.list_indexes()]
if index_name not in existing_indexes:
    print(f"Index '{index_name}' does not exist! Please create it in the console or via code.")
    exit()

index = pc.Index(index_name)

# 3. Connection Check
try:
    stats = index.describe_index_stats()
    print("Index Stats:", stats)
except Exception as e:
    print("Failed to connect to index:", e)
    exit()

# 4. FIXED: Cleaned up indentation
dummy_vector = [0.1] * 384 

try:
    index.upsert(
        vectors=[
            {
                "id": "test_id_1", 
                "values": dummy_vector, 
                "metadata": {"source": "manual_test"}
            }
        ]
    )
    print("Successfully connected and upserted a vector!")
except Exception as e:
    print(f"Upsert failed: {e}")