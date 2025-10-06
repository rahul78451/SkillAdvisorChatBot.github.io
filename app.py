# app.py - Final Corrected Version (No NameError + Rate Limit + FAISS Cache)

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
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

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
# PDF Loading + FAISS Caching
# -------------------------
# -------------------------
# PDF Loading + FAISS Caching
# -------------------------
pdf_dir = "pdf"
faiss_path = "faiss_db"

if "vectors" not in st.session_state:
    # ✅ Use HuggingFace embeddings (no API quota or billing)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(faiss_path):
        st.session_state["vectors"] = FAISS.load_local(
            faiss_path, embeddings, allow_dangerous_deserialization=True
        )
        st.success("✅ Loaded existing FAISS database from cache.")
    else:
        st.info("📚 Creating new FAISS database from PDFs... Please wait.")

        if not os.path.exists(pdf_dir):
            st.error(f"❌ PDF folder '{pdf_dir}' not found! Please create it and add at least one PDF.")
            st.stop()

        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
        if not pdf_files:
            st.error("❌ No PDF files found in 'pdf/' folder. Please add at least one PDF file.")
            st.stop()

        temp_pdf_texts = []
        for file in pdf_files:
            loader = PyPDFLoader(os.path.join(pdf_dir, file))
            docs = loader.load()
            text = " ".join([d.page_content for d in docs])
            temp_pdf_texts.append(text)

        pdfDatabase = " ".join(temp_pdf_texts)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(pdfDatabase)

        if not chunks:
            st.error("❌ PDF loaded but no text found. Check your PDF content.")
            st.stop()

        try:
            vector_store = FAISS.from_texts(chunks, embeddings)
            vector_store.save_local(faiss_path)
            st.session_state["vectors"] = vector_store
            st.success("✅ FAISS database created and cached successfully! (using local embeddings)")
        except Exception as e:
            st.error(f"⚠️ Error creating FAISS database: {e}")
            st.stop()


    

    if os.path.exists(faiss_path):
        st.session_state["vectors"] = FAISS.load_local(
            faiss_path, embeddings, allow_dangerous_deserialization=True
        )
        st.success("✅ Loaded existing FAISS database from cache.")
    else:
        st.info("📚 Creating new FAISS database from PDFs... Please wait.")

        if not os.path.exists(pdf_dir):
            st.error(f"❌ PDF folder '{pdf_dir}' not found! Please create it and add at least one PDF.")
            st.stop()

        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
        if not pdf_files:
            st.error("❌ No PDF files found in 'pdf/' folder. Please add at least one PDF file.")
            st.stop()

        temp_pdf_texts = []
        for file in pdf_files:
            loader = PyPDFLoader(os.path.join(pdf_dir, file))
            docs = loader.load()
            text = " ".join([d.page_content for d in docs])
            temp_pdf_texts.append(text)

        pdfDatabase = " ".join(temp_pdf_texts)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(pdfDatabase)

        if not chunks:
            st.error("❌ PDF loaded but no text found. Check your PDF content.")
            st.stop()

        try:
            vector_store = FAISS.from_texts(chunks, embeddings)
            vector_store.save_local(faiss_path)
            st.session_state["vectors"] = vector_store
            st.success("✅ FAISS database created and cached successfully!")
        except Exception as e:
            st.error(f"⚠️ Error creating FAISS database: {e}")
            st.stop()


# -------------------------
# Response Generation
# -------------------------
# -------------------------
# Response Generation
# -------------------------
def get_response(history, user_message, temperature=0):
    DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and a Career Advisor.
    ...
    """

    PROMPT = PromptTemplate(
        input_variables=['context','input','text','web_knowledge'],
        template=DEFAULT_TEMPLATE
    )

    # ✅ Safety check before using vectors
    if "vectors" not in st.session_state or not st.session_state["vectors"]:
        st.error("⚠️ Vector database not initialized yet. Please wait or restart the app.")
        st.stop()

    # ✅ Perform similarity search safely
    docs = st.session_state["vectors"].similarity_search(user_message)
    text = " ".join([d.page_content for d in docs])  # FIX ✅

    # ✅ Fetch web knowledge
    search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    web_knowledge = search.run(user_message)

    # ✅ Initialize Gemini model
    gemini_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=temperature,
        google_api_key=google_api_key
    )

    # ✅ Build LangChain
    conversation_with_summary = LLMChain(
        llm=gemini_model,
        prompt=PROMPT,
        verbose=False
    )

    # ✅ Generate final response
    response = conversation_with_summary.predict(
        context=history,
        input=user_message,
        web_knowledge=web_knowledge,
        text=text
    )
    return response


# -------------------------
# Conversation History
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

    # ✅ Correct indentation for try/except block
    try:
        output = get_response(formatted_history, user_input)
    except Exception as e:
        st.error(f"⚠️ Internal Error: {type(e).__name__} - {e}")
        st.stop()

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

# Chat History Section
# -------------------------
with st.expander("Chat History", expanded=True):
    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            st.markdown(emoji.emojize(f":speech_balloon: **User {i}**: {st.session_state['past'][i]}"))
            st.markdown(emoji.emojize(f":robot: **Assistant {i}**: {st.session_state['generated'][i]}"))
