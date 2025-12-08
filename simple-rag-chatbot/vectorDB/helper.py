import os
import requests
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
# --- MISSING IMPORT ADDED HERE ---
from pinecone import Pinecone 

load_dotenv()

# 1. Setup Hugging Face
hf_token = os.getenv("HF_TOKEN")
hf_client = InferenceClient(model="sentence-transformers/all-MiniLM-L6-v2", token=hf_token)

# 2. Setup Pinecone (THIS WAS MISSING)
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("INDEX_NAME", "mongo-sync-index")

if not pinecone_api_key:
    print("Error: PINECONE_API_KEY not found in .env")
    exit()

# Initialize the client and the index
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name) # <--- This defines the 'index' variable you were missing!

print(f"Using INDEX_NAME: {index_name}")

# 3. Setup API
API_URL = os.getenv("API_URL", "https://sih-2025-user.onrender.com/api/v1/alumni/all")
ID_FIELD = "_id"

print("Initializing...")
model = hf_client

print(f"Fetching data from {API_URL}...")

try:
    headers = {
        "x-analytics-api-key": "AnalyticsTopSecret",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    response = requests.get(API_URL, headers=headers)
    response.raise_for_status()
    
    data = response.json()
    
    if isinstance(data, dict) and 'data' in data:
        data = data['data']
    
    print(f"API returned {len(data)} records.")

except Exception as e:
    print(f"Error fetching API: {e}")
    exit()

# --- VECTORIZE & UPLOAD ---
batch_size = 100
batch_data = []
counter = 0

print("Starting vectorization...")

for item in data:
    text_parts = [
        f"Name: {item.get('name', '')}",
        f"Email: {item.get('email', '')}",
        f"User Type: {item.get('userType', '')}"
    ]
    if 'profileDetails' in item and item['profileDetails']:
        profile = item['profileDetails']
        if 'graduationYear' in profile:
            text_parts.append(f"Graduation Year: {profile['graduationYear']}")
        if 'skills' in profile and profile['skills']:
            text_parts.append(f"Skills: {', '.join(profile['skills'])}")
    
    text_content = " | ".join(text_parts)
    
    if text_content.strip():
        # --- CRITICAL FIX: Flatten the vector ---
        # feature_extraction often returns [[0.1, 0.2...]]. Pinecone needs [0.1, 0.2...]
        vector_response = model.feature_extraction(text_content).tolist()
        
        # If the response is a nested list (e.g. [[...]]), grab the first element
        if isinstance(vector_response[0], list):
             vector = vector_response[0]
        else:
             vector = vector_response

        record_id = str(item.get(ID_FIELD, "unknown_id"))
        record = {
            "id": record_id,
            "values": vector,
            "metadata": {
                "text": text_content,
                "source_api": "mongo_api"
            }
        }
        
        batch_data.append(record)
        
        if len(batch_data) >= batch_size:
            index.upsert(vectors=batch_data)
            counter += len(batch_data)
            print(f"Uploaded {counter} vectors...")
            batch_data = []

if batch_data:
    index.upsert(vectors=batch_data)
    counter += len(batch_data)

print(f"DONE: Processed {counter} records.")