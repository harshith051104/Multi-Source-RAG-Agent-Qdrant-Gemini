import gradio as gr
import os
import shutil
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    WebBaseLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import AIMessage, HumanMessage

# --- Environment and Helper Functions ---
load_dotenv()
google_api_key = os.environ.get("GOOGLE_API_KEY")

# --- Persistent User Authentication ---
CREDENTIALS_FILE = "user_credentials.json"

def load_credentials():
    """Loads user credentials from a JSON file."""
    if not os.path.exists(CREDENTIALS_FILE):
        default_credentials = {"user1": "password123", "user2": "password456"}
        with open(CREDENTIALS_FILE, 'w') as f:
            json.dump(default_credentials, f)
        return default_credentials
    with open(CREDENTIALS_FILE, 'r') as f:
        return json.load(f)

def save_credentials(credentials):
    """Saves user credentials to the JSON file."""
    with open(CREDENTIALS_FILE, 'w') as f:
        json.dump(credentials, f, indent=4)

USER_CREDENTIALS = load_credentials()

def check_login(username, password):
    """Validates user credentials against the loaded data."""
    return USER_CREDENTIALS.get(username) == password

# --- Other Helper Functions ---
def get_user_data_dir(username):
    base_dir = "user_data"
    user_dir = os.path.join(base_dir, username)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    return user_dir

def load_document(file_path):
    try:
        _, file_extension = os.path.splitext(file_path)
        loader_map = {
            '.pdf': PyPDFLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader,
            '.txt': TextLoader,
            '.csv': CSVLoader
        }
        loader = loader_map.get(file_extension.lower())
        if not loader:
            raise ValueError(f"Unsupported file type: {file_extension}")
        return loader(file_path).load()
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def sanitize_filename(filename):
    if not filename: return ""
    base_name, _ = os.path.splitext(filename)
    return "".join(c for c in base_name if c.isalnum() or c in ('_', '-')).rstrip().replace(' ', '_')

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def get_agent_executor(tools, api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2, google_api_key=api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. You must use the provided tools to answer questions. Answer based only on the context from the tools."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Gradio App UI and Logic ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="neutral"), title="Secure Multi-Source RAG Agent") as demo:
    username_state = gr.State("")
    agent_executor_state = gr.State(None)

    def handle_signup(new_username, new_password, confirm_password):
        if not new_username or not new_password:
            raise gr.Error("Username and password cannot be empty.")
        if new_password != confirm_password:
            raise gr.Error("Passwords do not match.")
        
        current_credentials = load_credentials()
        if new_username in current_credentials:
            raise gr.Error("Username already exists. Please choose another one.")
        
        current_credentials[new_username] = new_password
        save_credentials(current_credentials)
        
        global USER_CREDENTIALS
        USER_CREDENTIALS = current_credentials
        
        return "Sign up successful! Please go to the Login tab to log in."

    def handle_login(username, password):
        if check_login(username, password):
            initial_history = [(None, "Hello! How can I help you today? Please provide data sources in the sidebar first.")]
            return username, initial_history, gr.update(visible=False), gr.update(visible=True)
        else:
            raise gr.Error("Invalid username or password.")

    def handle_logout():
        return "", None, [], gr.update(visible=True), gr.update(visible=False)

    # --- MODIFIED: This function is corrected to avoid Qdrant lock conflicts ---
    def process_all_sources(username, uploaded_file_obj, web_urls_text):
        if not username: raise gr.Error("Login session expired.")
        if not google_api_key: raise gr.Error("GOOGLE_API_KEY not found.")
        
        tools = []
        user_data_dir = get_user_data_dir(username)
        status = "Scanning for existing Qdrant collections...\n"
        yield status
        
        # Each subdirectory in the user's data folder is a separate Qdrant collection
        if os.path.exists(user_data_dir):
            for collection_name in os.listdir(user_data_dir):
                collection_path = os.path.join(user_data_dir, collection_name)
                if os.path.isdir(collection_path):
                    try:
                        # **FIX:** The path now points to the unique collection directory
                        vectordb = QdrantVectorStore(
                            path=collection_path,
                            collection_name=collection_name,
                            embeddings=embeddings_model
                        )
                        retriever = vectordb.as_retriever()
                        tool_description = f"Search for info from the source: {collection_name}."
                        tools.append(create_retriever_tool(retriever, f"search_{collection_name}", tool_description))
                    except Exception as e:
                        print(f"Could not load collection '{collection_name}'. It might be invalid. Error: {e}")

        # Process newly uploaded file
        if uploaded_file_obj:
            original_filename = os.path.basename(uploaded_file_obj.name)
            collection_name = sanitize_filename(original_filename)
            # **FIX:** Define a unique path for this collection
            collection_path = os.path.join(user_data_dir, collection_name)

            status += f"Processing {original_filename} into collection '{collection_name}'...\n"
            yield status
            docs = load_document(uploaded_file_obj.name)
            if docs:
                documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
                # **FIX:** Use the unique collection_path for storage
                vectordb = QdrantVectorStore.from_documents(
                    documents=documents,
                    embedding=embeddings_model,
                    path=collection_path,
                    collection_name=collection_name,
                    force_recreate=True,
                )
                retriever = vectordb.as_retriever()
                tool_description = f"Search for info from the file: {original_filename}."
                tools.append(create_retriever_tool(retriever, f"search_{collection_name}", tool_description))
                if os.path.exists(uploaded_file_obj.name): os.remove(uploaded_file_obj.name)

        # Process Web URLs
        if web_urls_text and web_urls_text.strip():
            collection_name = "web_content_session"
            # **FIX:** Define a unique path for the web collection
            collection_path = os.path.join(user_data_dir, collection_name)
            status += "Processing web URLs into collection 'web_content_session'...\n"
            yield status

            urls = [url.strip() for url in web_urls_text.strip().split('\n') if url.strip()]
            all_web_docs = []
            for url in urls:
                try:
                    all_web_docs.extend(WebBaseLoader(url).load())
                except Exception as e:
                    print(f"Failed to load from URL '{url}': {e}")
            
            if all_web_docs:
                documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(all_web_docs)
                # **FIX:** Use the unique collection_path for storage
                vectordb_web = QdrantVectorStore.from_documents(
                    documents,
                    embeddings_model,
                    path=collection_path,
                    collection_name=collection_name,
                    force_recreate=True,
                )
                retriever_web = vectordb_web.as_retriever()
                tool_description = "Search for info from the provided web pages for this session."
                tools.append(create_retriever_tool(retriever_web, f"search_{collection_name}", tool_description))

        if not tools: raise gr.Error("No data sources were loaded. Please upload a file or provide a URL.")
        
        agent_executor = get_agent_executor(tools, google_api_key)
        status += f"\nProcessed all sources. {len(tools)} tools are active and ready."
        yield status
        agent_executor_state.value = agent_executor

    def handle_chat_message(prompt, history, agent_executor):
        if not agent_executor: raise gr.Error("Agent not ready. Please load data sources from the sidebar first.")
        
        langchain_history = []
        for user_msg, ai_msg in history:
            if user_msg: langchain_history.append(HumanMessage(content=user_msg))
            if ai_msg: langchain_history.append(AIMessage(content=ai_msg))

        history.append((prompt, None))
        yield history
        try:
            response = agent_executor.invoke({"input": prompt, "chat_history": langchain_history})
            history[-1] = (prompt, response["output"])
            yield history
        except Exception as e:
            history[-1] = (prompt, f"An error occurred: {e}")
            yield history

    # --- UI Definition ---
    with gr.Group(visible=True) as login_group:
        gr.Markdown("# Welcome to the Multi-Source RAG Agent")
        with gr.Tabs():
            with gr.TabItem("Login"):
                gr.Markdown("## 🔐 Login")
                username_input = gr.Textbox(label="Username", placeholder="Enter your username")
                password_input = gr.Textbox(label="Password", type="password", placeholder="Enter your password")
                login_button = gr.Button("Login", variant="primary")
            with gr.TabItem("Sign Up"):
                gr.Markdown("## 📝 Sign Up")
                new_username_input = gr.Textbox(label="New Username", placeholder="Choose a username")
                new_password_input = gr.Textbox(label="New Password", type="password", placeholder="Choose a password")
                confirm_password_input = gr.Textbox(label="Confirm Password", type="password", placeholder="Confirm your password")
                signup_button = gr.Button("Sign Up", variant="primary")
                signup_status = gr.Markdown()

    with gr.Group(visible=False) as main_app_group:
        gr.Markdown("# 📄🤖 Multi-Source RAG Agent (Qdrant & Gemini)")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Data Sources")
                logout_button = gr.Button("Logout")
                if not google_api_key: gr.Warning("GOOGLE API key not found.")
                uploaded_file = gr.File(label="Upload a Document", file_types=['.pdf', '.docx', '.txt', '.csv'], type="filepath")
                web_urls_input = gr.Textbox(label="Enter Web Page URLs (one per line)", lines=4, placeholder="https://example.com\n...")
                process_button = gr.Button("Load & Process All Sources", variant="primary")
                status_display = gr.Markdown("Status: Waiting for sources...")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat", bubble_full_width=False, height=600, avatar_images=(None, "https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Qdrant_logo_main.eff1895.svg/2560px-Qdrant_logo_main.eff1895.svg.png"))
                prompt_input = gr.Textbox(label="Ask a question...", show_label=False, placeholder="Type your question here...")

    # --- Event Wiring ---
    login_button.click(handle_login, [username_input, password_input], [username_state, chatbot, login_group, main_app_group])
    signup_button.click(handle_signup, [new_username_input, new_password_input, confirm_password_input], [signup_status])
    logout_button.click(handle_logout, [], [username_state, agent_executor_state, chatbot, login_group, main_app_group])
    process_button.click(process_all_sources, [username_state, uploaded_file, web_urls_input], [status_display]).then(lambda: agent_executor_state.value, [], [agent_executor_state])
    prompt_input.submit(handle_chat_message, [prompt_input, chatbot, agent_executor_state], [chatbot]).then(lambda: "", outputs=[prompt_input])

if __name__ == "__main__":
    demo.launch(debug=True)