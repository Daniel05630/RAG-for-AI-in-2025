# RAG Application with LangGraph Agent

This project is a Retrieval-Augmented Generation (RAG) application that uses a LangGraph-powered agent to answer questions based on the content of uploaded PDF files. It provides a RESTful API for file management and chat interactions, along with a simple web interface.

## Features

- **PDF File Upload and Processing**: Ingest PDF files for content extraction.
- **Vector Embeddings**: Creates vector embeddings for document chunks using HuggingFace sentence transformers.
- **Vector Store**: Manages and stores embeddings in a vector database.
- **RAG Agent**: Utilizes a LangGraph agent to process user queries, retrieve relevant information, and generate answers.
- **Chat API**: Provides endpoints for asking questions and managing chat sessions.
- **File Management API**: Endpoints to upload, list, and delete files.
- **Web Interface**: A simple HTML frontend for interacting with the application.

## Tech Stack

- **Backend**: FastAPI, Uvicorn
- **LLM Orchestration**: LangChain, LangGraph
- **Embeddings**: HuggingFace Sentence Transformers
- **LLM**: Groq
- **Vector Database**: Pinecone
- **Data Handling**: Pydantic, PyPDF2
- **Frontend**: HTML, JavaScript

## API Endpoints

All endpoints are prefixed with `/api/v1`.

### File Management

- `POST /files/upload`: Upload a PDF file.
- `GET /files`: List all uploaded files.
- `GET /files/{file_id}`: Get information about a specific file.
- `DELETE /files/{file_id}`: Delete a file and its associated embeddings.
- `PUT /files/{file_id}`: Update an existing file.

### Chat

- `POST /chat`: Submit a question to the RAG agent.
- `GET /chat/sessions/{session_id}`: Retrieve the chat history for a session.
- `DELETE /chat/sessions/{session_id}`: Clear the chat history for a session.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**
    Create a `.env` file in the project root and add the necessary API keys and configuration settings. A `.env.example` file should be provided to show the required variables.

    ```
    # .env file
    GROQ_API_KEY="your_groq_api_key"
    PINECONE_API_KEY="your_pinecone_api_key"
    # Add other configuration as needed
    ```

5.  **Run the application:**
    ```bash
    uvicorn main:app --reload
    ```
    The application will be available at `http://127.0.0.1:8000`.

## Project Structure

```
.
├── api/                # API route definitions
│   ├── routes_chat.py
│   └── routes_files.py
├── core/               # Core components (e.g., configuration)
│   └── config.py
├── langgraph_agent/    # LangGraph agent implementation
│   ├── agent.py
│   ├── state.py
│   └── tools.py
├── services/           # Business logic services
│   ├── data_ingestion_service.py
│   ├── embeddings_service.py
│   └── vectordb_service.py
├── static/             # Frontend files
│   └── index.html
├── utils/              # Utility functions (e.g., logger)
│   └── logger.py
├── .env                # Environment variables
├── main.py             # FastAPI application entry point
├── requirements.txt    # Project dependencies
└── README.md           # This file
```
