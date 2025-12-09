import os
from dotenv import load_dotenv
import chainlit as cl
from groq import AsyncGroq  # Correct Import
from pinecone import Pinecone
from huggingface_hub import InferenceClient

load_dotenv()

# --- 1. SETUP HUGGING FACE & PINECONE ---
hf_token = os.getenv("HF_TOKEN")
# Renamed to 'hf_client' to avoid confusion with Groq client later
hf_client = InferenceClient(model="sentence-transformers/all-MiniLM-L6-v2", token=hf_token)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "mongo-sync-index")

SYSTEM_PROMPT = """You are a retrieval-augmented chatbot for alumni information.
Rules:
1. Use the retrieved context to answer if relevant.
2. Keep answers short and direct.
3. If context is missing, use general knowledge or say "I don't have enough information".
4. Do not invent facts.
"""

print("Initializing Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
print("Done!")

def retrieve_from_pinecone(query: str, top_k: int = 47):
    """Retrieve relevant documents from Pinecone"""
    # 1. Generate Embedding
    response = hf_client.feature_extraction(query)
    
    # 2. Fix formatting (Numpy -> List)
    if hasattr(response, 'tolist'):
        response = response.tolist()
        
    # 3. Flatten list if nested (API often returns [[0.1, 0.2...]])
    if isinstance(response[0], list):
        query_vector = response[0]
    else:
        query_vector = response

    # 4. Query Pinecone
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    relevant_docs = []
    for match in results['matches']:
        if match['score'] > 0.3: 
            relevant_docs.append(match['metadata'].get('text', ''))
    return relevant_docs


@cl.on_chat_start
async def on_chat_start():
    # --- FIX 1: Correct Class Name ---
    client = AsyncGroq(api_key=GROQ_API_KEY)
    cl.user_session.set("client", client)

    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("chat_history", chat_history)

    await cl.Message(content="Welcome! How may I assist you today?").send()


@cl.on_message
async def main(message: cl.Message):
    client: AsyncGroq = cl.user_session.get("client")
    chat_history: list = cl.user_session.get("chat_history")

    try:
        chat_history.append({"role": "user", "content": message.content})

        # Run retrieval in a separate thread so it doesn't block the UI
        relevant_documents = await cl.make_async(retrieve_from_pinecone)(message.content, top_k=3)
        
        if relevant_documents:
            relevant_info = "\n\n".join(relevant_documents)
            chat_history.append(
                {"role": "system", "content": f"Relevant Documents: \n{relevant_info}"}
            )

        # --- FIX 2: Add 'await' here ---
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=chat_history
        )
        
        reply = response.choices[0].message.content

        chat_history.append({"role": "assistant", "content": reply})
        cl.user_session.set("chat_history", chat_history)

        await cl.Message(content=reply).send()

    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()