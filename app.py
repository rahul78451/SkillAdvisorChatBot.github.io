# app.py - Fully Hardened Version (Resilient to API/Index Errors)

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
import google.generativeai as genai

# -------------------------
# Fix for Windows Event Loop
# -------------------------
if threading.current_thread() is threading.main_thread():
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
else:
    asyncio.set_event_loop(asyncio.new_event_loop())

# -------------------------
# API Keys from Streamlit Secrets
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

# -------------------------
# FAISS Database Caching and Creation
# -------------------------
pdf_dir = 'pdf'
faiss_path = "faiss_db"

# Initialize embeddings once, as it's used for both loading and creation
@st.cache_resource
def get_embeddings(key):
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=key
    )

embeddings = get_embeddings(google_api_key)

# 🔑 FIX: Renamed 'embeddings' to '_embeddings' to make it an unhashed argument.
@st.cache_resource(show_spinner=False)
def get_vector_store(faiss_path, pdf_dir, _embeddings):
    """
    Attempts to load the vector store from cache or create it from PDFs.
    Uses st.cache_resource to ensure it only runs once per session/change.
    """
    st.warning("Attempting to load or create database. Check API Quotas if this fails.")
    
    # 1. Try loading FAISS cache
    if os.path.exists(faiss_path):
        try:
            # Use _embeddings
            vector_store = FAISS.load_local(faiss_path, _embeddings, allow_dangerous_deserialization=True)
            st.success("✅ Loaded existing database from cache.")
            return vector_store
        except Exception as e:
            st.error(f"⚠️ Could not load FAISS cache: {e}. Attempting to re-create.")
    
    # 2. Create database
    vector_store = None
    temp_pdf_texts = []
    
    with st.spinner("📚 Creating a Database..."):
        try:
            # Load documents
            for file in os.listdir(pdf_dir):
                if file.endswith('.pdf'):
                    loader = PyPDFLoader(os.path.join(pdf_dir, file))
                    documents = loader.load()
                    text = " ".join([doc.page_content for doc in documents])
                    temp_pdf_texts.append(text)
        except FileNotFoundError:
            st.error(f"Error: The directory '{pdf_dir}' was not found. Cannot create database.")
            return None
        except Exception as e:
            st.error(f"Error loading PDF files: {e}. Cannot create database.")
            return None

        if not temp_pdf_texts:
            st.error("❌ No text could be loaded from the PDF directory. Cannot create database.")
            return None

        # Combine all text and split into chunks
        pdfDatabase = " ".join(temp_pdf_texts)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(pdfDatabase)

        if not chunks:
            st.error("❌ Document processing failed: No text chunks were generated.")
            return None

        # -------------------------
        # Batching + Rate Limit Handling
        # -------------------------
        BATCH_SIZE = 25
        DELAY_SECONDS = 5
        total_chunks = len(chunks)

        status_placeholder = st.empty()

        for i in range(0, total_chunks, BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            status_placeholder.text(f"Processing batch {i//BATCH_SIZE + 1} of {total_chunks//BATCH_SIZE + 1}...")

            try:
                if vector_store is None:
                    # Initializes the vector store with the first batch
                    # Use _embeddings
                    vector_store = FAISS.from_texts(batch, _embeddings)
                else:
                    # Adds subsequent batches
                    vector_store.add_texts(batch)

                if i + BATCH_SIZE < total_chunks:
                    time.sleep(DELAY_SECONDS)  # prevent hitting API rate limit

            except Exception as e:
                # Catches all API/Embedding errors, including Quota Exceeded and IndexError
                if "429" in str(e) or "quota" in str(e).lower():
                    st.error("🚨 Quota Exceeded. Database creation halted.")
                else:
                    st.error(f"Unexpected error during FAISS creation: {e}")
                
                status_placeholder.empty()
                return None # Return None on failure
        
        status_placeholder.empty()

        if vector_store:
            vector_store.save_local(faiss_path)
            st.success("✅ Database creation completed and cached!")
            return vector_store
        
        # Should not happen, but catches final failure
        st.error("❌ Failed to create vector store at the end of processing.")
        return None

# *** FIX 4: Call the cached function and set session state only if needed ***
# NOTE: The call here is correct because 'embeddings' is passed as the third argument.
if "vectors" not in st.session_state:
    with st.spinner("Initializing system..."):
        st.session_state["vectors"] = get_vector_store(faiss_path, pdf_dir, embeddings)

# Display readiness status
if st.session_state["vectors"] is not None:
    st.info("Vector database is ready to answer questions.")
else:
    # This replaces the initial red error box that was hardcoded
    st.error("FAISS Database is not loaded. Please fix the error messages above (Quota, files, or Index) and rerun.")


# -------------------------
# Response Generation (Keep the function as is, it's correct)
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
        input_variables=['context','input','text','web_knowledge'],
        template=DEFAULT_TEMPLATE
    )

    # This line is now protected by the check in the UI logic
    docs = st.session_state["vectors"].similarity_search(user_message)  

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

# -------------------------
# Conversation History (get_history is correct)
# -------------------------
def get_history(history_list):
    history = ''
    for message in history_list:
        if message['role'] == 'user':
            history += 'input ' + message['content'] + '\n'
        elif message['role'] == 'assistant':
            history += 'output ' + message['content'] + '\n'
    return history

# -------------------------
# Streamlit UI
# -------------------------
def get_text():
    # Use st.form for a better button/input interaction loop
    with st.sidebar.form(key='chat_form', clear_on_submit=True):
        input_text = st.text_input("You: ", "Hello, how are you?", key="input")
        submit_button = st.form_submit_button('Send')
    
    if submit_button:
        return input_text
    return None

if "past" not in st.session_state:
    st.session_state["past"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []

user_input = get_text()

# *** FIX 5: Explicitly check if the vector store is ready before running chat logic ***
if user_input and st.session_state["vectors"] is not None:
    # Run chat logic only if vector store is ready
    user_history = list(st.session_state["past"])
    bot_history = list(st.session_state["generated"])

    combined_history = []
    for user_msg, bot_msg in zip_longest(user_history, bot_history):
        if user_msg is not None:
            combined_history.append({'role': 'user', 'content': user_msg})
        if bot_msg is not None:
            combined_history.append({'role': 'assistant', 'content': bot_msg})

    formatted_history = get_history(combined_history)
    output = get_response(formatted_history, user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
elif user_input and st.session_state["vectors"] is None:
    # This handles the second error box shown in your image
    st.error("Cannot send message. The FAISS Database is not loaded due to a previous error (Quota/File/Index).")


with st.expander("Chat History", expanded=True):
    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            st.markdown(emoji.emojize(f":speech_balloon: **User {i}**: {st.session_state['past'][i]}"))
            st.markdown(emoji.emojize(f":robot: **Assistant {i}**: {st.session_state['generated'][i]}"))
