import os
from getpass import getpass
from pathlib import Path

import chainlit as cl
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
import os
api_key = os.getenv("API-KEY")


os.environ["GROQ_API_KEY"] =api_key

SYSTEM_PROMPT = """You are a retrieval-augmented chatbot.

Always follow these rules:
1. Use the retrieved context to answer the user, but only if it is relevant to the query.
2. Keep your answers short, direct, and clear.
3. If the context does not contain useful or relevant information for the query, answer using your general knowledge.
4. If the question requires factual details that are missing from both the context and your general knowledge, say: "I don't have enough information in my documents."
5. Do not invent facts that are not present in the context.
6. Never mention that you were given context or retrieved documents.
"""

# RAG retriever tool
loader = PyPDFLoader(Path(__file__).parent / "Career_Advice_Software_AI_ML.pdf")
pages = loader.load_and_split()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(pages)
print("Indexing documents...")
vector = FAISS.from_documents(documents, HuggingFaceEmbeddings())
print("done!")
retriever = vector.as_retriever()


@cl.on_chat_start
async def on_chat_start():
    # LLM client
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    cl.user_session.set("client", client)

    # Chat history
    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("chat_history", chat_history)

    cl.user_session.set("retriever", retriever)

    welcome_message = (
        "Welcome!"
        " How may I assist you?"
    )

    chat_history.append({"role": "assistant", "content": welcome_message})
    await cl.Message(content=welcome_message).send()


@cl.on_message
async def main(message: cl.Message):
    client: Groq = cl.user_session.get("client")
    chat_history: list = cl.user_session.get("chat_history")
    retriever: VectorStoreRetriever = cl.user_session.get("retriever")

    try:
        chat_history.append({"role": "user", "content": message.content})

        #retrieving doc for response 
        relevant_documents = retriever.invoke(message.content)
        if len(relevant_documents) > 0:
            relevant_info = "\n\n".join(
                map(lambda doc: doc.page_content, relevant_documents)
            )
            chat_history.append(
                {"role": "system", "content": f"Relevant Documents: \n{relevant_info}"}
            )
        #llm's response
        response = client.chat.completions.create(model="openai/gpt-oss-120b", messages=chat_history)
        reply = response.choices[0].message.content

        chat_history.append({"role": "assistant", "content": reply})
        cl.user_session.set("chat_history", chat_history)

        #sending lls's response
        await cl.Message(content=reply if reply is not None else "").send()

    except Exception as e:
        # Handle any exceptions that occur during the API request
        await cl.Message(content=f"An error occurred: {str(e)}").send()


@cl.on_stop
async def on_stop():
    # Clear the chat history when the session ends
    cl.user_session.set("chat_history", [])
