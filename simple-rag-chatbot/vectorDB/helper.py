import os
import requests
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
API_URL = os.getenv("API_URL", "https://sih-2025-user.onrender.com/api/v1/alumni/all")
API_KEY = os.getenv("API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "mongo-sync-index")

# Debug: Print to verify
print(f"Using INDEX_NAME: {INDEX_NAME}")

ID_FIELD = "_id"

# --- INITIALIZATION ---
print("Initializing...")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- FETCH DATA FROM API ---
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
    
    # Handle nested response structure
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
    # Create text from available fields
    text_parts = [
        f"Name: {item.get('name', '')}",
        f"Email: {item.get('email', '')}",
        f"User Type: {item.get('userType', '')}"
    ]
    
    # Add profile details if available
    if 'profileDetails' in item and item['profileDetails']:
        profile = item['profileDetails']
        if 'graduationYear' in profile:
            text_parts.append(f"Graduation Year: {profile['graduationYear']}")
        if 'skills' in profile and profile['skills']:
            text_parts.append(f"Skills: {', '.join(profile['skills'])}")
    
    text_content = " | ".join(text_parts)
    
    if text_content.strip():
        vector = model.encode(text_content).tolist()
        
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