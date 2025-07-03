# Secure Multi-Source RAG Agent with Gradio, Qdrant & Gemini

This project implements a sophisticated, secure, and user-friendly RAG (Retrieval-Augmented Generation) agent. It features a Gradio web interface, persistent user authentication, and personalized, multi-source data handling powered by LangChain, Qdrant, and Google's Gemini model.

The key innovation is its ability to create and manage separate, persistent vector stores for each user and data source. This ensures that data is not only secure but also that the AI's responses are precisely tailored to the specific context provided by the user's documents.

![A screenshot of the Gradio interface showing the chat window and the data source upload sidebar.]("C:\Users\sriha\OneDrive\Pictures\Screenshots 1\Screenshot 2025-07-03 230418.png")

## ✨ Features

- **🔒 Secure User Authentication**: A persistent login/signup system ensures that each user's data and chat history are kept private. Credentials are saved in a `user_credentials.json` file.
- **👤 User-Specific Data Storage**: Each user has a dedicated directory (`user_data/<username>`) where their data collections are stored, preventing data crossover between users.
- **📚 Multi-Source Data Ingestion**: Supports a wide array of data types:
    - PDF files (`.pdf`)
    - Word documents (`.doc`, `.docx`)
    - Text files (`.txt`)
    - CSV files (`.csv`)
    - Live web page content (via URLs)
- **🧠 Persistent Vector Stores**: Each uploaded document or web source is processed into its own persistent **Qdrant** vector collection. This avoids reprocessing and allows the agent to access a growing library of user-specific knowledge across sessions.
- **🤖 Intelligent RAG Agent**: Utilizes **LangChain** and Google's powerful **Gemini** model to create a conversational agent that can reason over the user's documents. The agent dynamically creates a set of retriever tools—one for each data source—to find the most relevant information.
- **💬 Interactive Chat Interface**: A clean and intuitive chat UI built with **Gradio** allows for real-time interaction, complete with chat history.
- **⚙️ Dynamic Tool Creation**: The agent's tools are generated on-the-fly based on the data sources the user has provided, making it highly adaptable.

## 🚀 How It Works

1.  **Authentication**: The user logs in or signs up. The application authenticates them against a stored JSON file.
2.  **Data Loading**: Once logged in, the user can upload documents or provide URLs.
3.  **Vectorization & Storage**: For each data source, the application:
    - Sanitizes the source name to create a valid collection name (e.g., `my_report.pdf` becomes `my_report`).
    - Creates a dedicated subdirectory under `user_data/<username>/<collection_name>`.
    - Loads and splits the document content into chunks.
    - Uses **HuggingFace sentence-transformers** to generate embeddings for each chunk.
    - Stores these embeddings in a local, on-disk **Qdrant** vector store within the dedicated subdirectory.
4.  **Agent Assembly**: The application scans the user's data directory, loading all existing Qdrant collections. It creates a unique `retriever_tool` for each one.
5.  **Chat Interaction**:
    - The user asks a question.
    - The **LangChain AgentExecutor** (powered by Gemini) receives the prompt.
    - The agent intelligently decides which of its retriever tools is most likely to contain the answer.
    - It queries the relevant Qdrant collection(s), retrieves the context, and uses it to generate a precise, source-based answer.

This architecture prevents "context bleed" between documents and allows the agent to cite which source its information is coming from, leading to more accurate and trustworthy responses.

## 🛠️ Setup and Installation

### Prerequisites

- Python 3.8+
- A Google API Key with the "Generative Language API" enabled.

### 1. Clone the Repository

```bash
git clone <your-repo-link>
cd <your-repo-name>
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

Create a file named `requirements.txt` in your project folder, paste the contents from the section below into it, and run:

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a file named `.env` in the root directory of the project and add your Google API key:

```
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
```

### 5. Run the Application

Launch the Gradio app with the following command (replace `app.py` with the name of your Python script):

```bash
python app.py
```

The application will be available at a local URL shown in your terminal (e.g., `http://127.0.0.1:7860`).

---
