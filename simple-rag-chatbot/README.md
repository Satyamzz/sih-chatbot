
# Simple RAG Chatbot

An educational chatbot project demonstrating Retrieval-Augmented Generation (RAG), designed as a reference for developers exploring document-based AI assistants.

## Overview

This project showcases how to combine document retrieval with OpenAI's language models to answer user queries using relevant content from uploaded documents. It serves as a practical reference for developers interested in building AI-powered assistants.

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aporia-ai/simple-rag-chatbot.git
   cd simple-rag-chatbot
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

4. **Set the OpenAI API Key**:
   ```bash
   export OPENAI_API_KEY=your_api_key
   ```

## Usage

Run the chatbot using Chainlit:
```bash
chainlit run src/simple_rag_chatbot.py
```

## License

MIT License. See LICENSE file for details.
