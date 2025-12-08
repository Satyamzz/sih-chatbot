import os
from dotenv import load_dotenv
import chainlit as cl
from groq import Groq
from pinecone import Pinecone
from huggingface_hub import InferenceClient
load_dotenv()

hf_token=os.getenv("HF_TOKEN")

client = InferenceClient(model="sentence-transformers/all-MiniLM-L6-v2", token=hf_token)


# Environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "mongo-sync-index")

SYSTEM_PROMPT = """You are a retrieval-augmented chatbot for alumni information.

Always follow these rules:
1. Use the retrieved context to answer the user, but only if it is relevant to the query.
2. Keep your answers short, direct, and clear .
3. If the context does not contain useful or relevant information for the query, answer using your general knowledge.
4. If the question requires factual details that are missing from both the context and your general knowledge, say: "I don't have enough information in my documents."
5. Do not invent facts that are not present in the context.
6. Never mention that you were given context or retrieved documents.
"""

# Initialize Pinecone and model
print("Initializing Pinecone and embedding model...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embedding_model = client
print("Done!")


def retrieve_from_pinecone(query: str, top_k: int = 3):
    """Retrieve relevant documents from Pinecone"""
    query_vector = embedding_model.feature_extraction(query).tolist()
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
    # LLM client
    client = Groq(api_key=GROQ_API_KEY)
    cl.user_session.set("client", client)

    # Chat history
    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("chat_history", chat_history)

    welcome_message = (
        "Welcome to the Alumni Information Chatbot! "
        "How may I assist you?"
    )

    chat_history.append({"role": "assistant", "content": welcome_message})
    await cl.Message(content=welcome_message).send()


@cl.on_message
async def main(message: cl.Message):
    client: Groq = cl.user_session.get("client")
    chat_history: list = cl.user_session.get("chat_history")

    try:
        chat_history.append({"role": "user", "content": message.content})

        # Retrieve relevant documents from Pinecone
        relevant_documents = retrieve_from_pinecone(message.content, top_k=3)
        
        if len(relevant_documents) > 0:
            relevant_info = "\n\n".join(relevant_documents)
            chat_history.append(
                {"role": "system", "content": f"Relevant Documents: \n{relevant_info}"}
            )

        # Get LLM's response
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=chat_history
        )
        reply = response.choices[0].message.content

        chat_history.append({"role": "assistant", "content": reply})
        cl.user_session.set("chat_history", chat_history)

        # Send LLM's response
        await cl.Message(content=reply if reply is not None else "").send()

    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()


@cl.on_stop
async def on_stop():
    cl.user_session.set("chat_history", [])