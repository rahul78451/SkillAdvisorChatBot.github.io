import os
import time
import asyncio
import threading
from itertools import zip_longest

import streamlit as st
import emoji

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SerpAPIWrapper
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
import google.generativeai as genai

# Disable Streamlit watchdog file watcher (Good practice for heavy IO)
os.environ["STREAMLIT_WATCHDOG"] = "false"

# -------------------------
# Event Loop Fix for Windows
# -------------------------
if threading.current_thread() is threading.main_thread():
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass
else:
    asyncio.set_event_loop(asyncio.new_event_loop())

# -------------------------
# Streamlit Secret Keys
# -------------------------
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Google API key not found in secrets.toml. Please add it to .streamlit/secrets.toml")
    st.stop()

if "SERPAPI_API_KEY" not in st.secrets:
    st.error("SerpAPI key not found in secrets.toml. Please add it to .streamlit/secrets.toml")
    st.stop()

google_api_key = st.secrets["GOOGLE_API_KEY"]
serpapi_api_key = st.secrets["SERPAPI_API_KEY"]

genai.configure(api_key=google_api_key)

st.title(f"Career Advisor Chatbot {emoji.emojize(':robot:')}")

# Directory and FAISS DB Path
pdf_dir = 'pdf'
faiss_path = "faiss_db"

# ==========================================================
# 🚀 FIX: Corrected PDF Loading, Chunking, and FAISS Creation
# ==========================================================
if "vectors" not in st.session_state:
    with st.spinner("Creating or loading database..."):
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )

        # 1. Try loading from cached FAISS DB first
        if os.path.exists(faiss_path):
            try:
                st.session_state["vectors"] = FAISS.load_local(
                    faiss_path, embeddings, allow_dangerous_deserialization=True
                )
                st.success("✅ Loaded cached FAISS database.")
                
            except Exception as e:
                # If loading fails, initialize an empty FAISS store
                st.warning(f"⚠️ Failed to load cache: {e}. Attempting to rebuild.")
                st.session_state["vectors"] = FAISS.from_texts([], embeddings)

        
        # 2. Build FAISS from PDFs if it couldn't be loaded or path didn't exist
        if "vectors" not in st.session_state or len(st.session_state["vectors"].docstore._dict) == 0:
            
            all_documents = []
            try:
                # Load all PDF files into LangChain Document objects
                for file_name in os.listdir(pdf_dir):
                    if file_name.endswith('.pdf'):
                        file_path = os.path.join(pdf_dir, file_name)
                        loader = PyPDFLoader(file_path)
                        all_documents.extend(loader.load())
                        
            except FileNotFoundError:
                st.error(f"Error: The directory '{pdf_dir}' was not found. Please create it and add PDFs.")
                st.session_state["vectors"] = FAISS.from_texts([], embeddings)
                st.stop()
            except Exception as e:
                st.error(f"Error loading PDFs: {e}")
                st.session_state["vectors"] = FAISS.from_texts([], embeddings)
                st.stop()

            if all_documents:
                # Chunk the documents
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(all_documents) # chunks is a list of Document objects

                if not chunks:
                    st.warning("⚠️ No text was extracted from the PDFs.")
                    st.session_state["vectors"] = FAISS.from_texts([], embeddings)
                else:
                    # Create embeddings and FAISS index using from_documents
                    # This is the correct method when starting with LangChain Document objects
                    vector_store = None
                    BATCH_SIZE = 25 
                    DELAY_SECONDS = 5
                    
                    try:
                        # Initial creation of the vector store
                        first_batch = chunks[:BATCH_SIZE]
                        vector_store = FAISS.from_documents(first_batch, embeddings)
                        
                        # Add remaining chunks in batches
                        for i in range(BATCH_SIZE, len(chunks), BATCH_SIZE):
                            batch = chunks[i:i + BATCH_SIZE]
                            vector_store.add_documents(batch) 
                            
                            # Pause to respect API rate limits
                            if i + BATCH_SIZE < len(chunks):
                                time.sleep(DELAY_SECONDS)

                        # Save and store the vector store
                        vector_store.save_local(faiss_path)
                        st.session_state["vectors"] = vector_store
                        st.success("✅ Database created, saved, and ready!")
                        
                    except Exception as e:
                        if "429" in str(e):
                            st.error("🚨 Quota Exceeded during embedding. Try again later.")
                        else:
                            st.error(f"Unexpected error during embedding: {e}")
                        st.session_state["vectors"] = FAISS.from_texts([], embeddings)
                        st.stop()

            else:
                st.session_state["vectors"] = FAISS.from_texts([], embeddings)
                st.warning("⚠️ No PDFs found or no content extracted. Using empty vector store.")


# -------------------------
# Response Generation
# -------------------------
def get_response(history, user_message, temperature=0):
    DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and a Career Advisor. 
The Advisor guides the user regarding jobs, interests, and domain selection decisions. 
It follows the previous conversation.

Relevant pieces of previous conversation:
{context},

Useful information from career guidance books:
{text},

Useful information from Web:
{web_knowledge},

Current conversation:
Human: {input}
Career Expert:"""

    PROMPT = PromptTemplate(
        input_variables=['context','input','text','web_knowledge'], template=DEFAULT_TEMPLATE
    )

    # Safety: Ensure vectors exist
    docs = []
    if "vectors" in st.session_state and st.session_state["vectors"]:
        docs = st.session_state["vectors"].similarity_search(user_message)
    # Note: If docs is empty, the LLM will only rely on web_knowledge/context.

    search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    web_knowledge = search.run(user_message)

    gemini_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=temperature,
        google_api_key=google_api_key
    )

    conversation_with_summary = LLMChain(
        llm=gemini_model,
        prompt=PROMPT,
        verbose=False
    )

    response = conversation_with_summary.predict(
        context=history,
        input=user_message,
        web_knowledge=web_knowledge,
        text=docs
    )
    return response

def get_history(history_list):
    history = ''
    for message in history_list:
        if message['role']=='user':
            history += 'input ' + message['content'] + '\n'
        elif message['role']=='assistant':
            history += 'output ' + message['content'] + '\n'
    return history

# -------------------------
# Streamlit UI
# -------------------------
def get_text():
    input_text = st.sidebar.text_input("You: ", "Hello, how are you?", key="input")
    if st.sidebar.button('Send'):
        return input_text
    return None

if "past" not in st.session_state:
    st.session_state["past"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []

user_input = get_text()

if user_input:
    user_history = list(st.session_state["past"])
    bot_history = list(st.session_state["generated"])

    combined_history = []
    for user_msg, bot_msg in zip_longest(user_history, bot_history):
        if user_msg is not None:
            combined_history.append({'role': 'user', 'content': user_msg})
        if bot_msg is not None:
            combined_history.append({'role': 'assistant', 'content': bot_msg})

    formatted_history = get_history(combined_history)
    
    # Check if a vector store was successfully initialized before calling the response function
    if "vectors" in st.session_state:
        output = get_response(formatted_history, user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
    else:
        st.error("Cannot process query: Vector database failed to load or create.")


with st.expander("Chat History", expanded=True):
    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            st.markdown(emoji.emojize(f":speech_balloon: **User {i}**: {st.session_state['past'][i]}"))
            st.markdown(emoji.emojize(f":robot: **Assistant {i}**: {st.session_state['generated'][i]}"))
