import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()
import os
api_key = os.getenv("pinecode-api")

# 1. Initialize the Client
# Replace with your actual API key
pc = Pinecone(api_key=api_key)

# 2. define index name
index_name = "mongo-sync-index"

import time 
index_name = "mongo-sync-index"
index = pc.Index(index_name)
print(index.describe_index_stats())

# 4. Test Upsert (Insert data)
# Format: (id, vector_values, metadata)
# This vector must have the exact same length as the 'dimension' you set above.
dummy_vector = [0.1] * 384 

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