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
# PDF Loading + FAISS Caching
# -------------------------
pdf_dir = 'pdf'
faiss_path = "faiss_db"

if "vectors" not in st.session_state:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )

    # Try loading FAISS cache first
    if os.path.exists(faiss_path):
        st.session_state["vectors"] = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        st.success("✅ Loaded existing database from cache.")
    else:
        temp_pdf_texts = []
        with st.spinner("📚 Creating a Database..."):
            try:
                for file in os.listdir(pdf_dir):
                    if file.endswith('.pdf'):
                        loader = PyPDFLoader(os.path.join(pdf_dir, file))
                        documents = loader.load()
                        text = " ".join([doc.page_content for doc in documents])
                        temp_pdf_texts.append(text)
            except FileNotFoundError:
                st.error(f"Error: The directory '{pdf_dir}' was not found.")
                st.stop()

            # Combine all text and split into chunks
            pdfDatabase = " ".join(temp_pdf_texts)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text(pdfDatabase)

            # -------------------------
            # Batching + Rate Limit Handling
            # -------------------------
            BATCH_SIZE = 25
            DELAY_SECONDS = 5
            vector_store = None
            total_chunks = len(chunks)

            for i in range(0, total_chunks, BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE]
                try:
                    if vector_store is None:
                        vector_store = FAISS.from_texts(batch, embeddings)
                    else:
                        vector_store.add_texts(batch)

                    if i + BATCH_SIZE < total_chunks:
                        time.sleep(DELAY_SECONDS)  # prevent hitting API rate limit

                except Exception as e:
                    if "429" in str(e):
                        st.error("🚨 Quota Exceeded. Using cached database if available.")
                        if os.path.exists(faiss_path):
                            vector_store = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
                        break
                    else:
                        st.error(f"Unexpected error: {e}")
                        st.stop()

            if vector_store:
                vector_store.save_local(faiss_path)
                st.session_state["vectors"] = vector_store
                st.success("✅ Database creation completed and cached!")

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

    docs = st.session_state["vectors"].similarity_search(user_message)
    text = " ".join([d.page_content for d in docs])  # FIX ✅

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
        text=text   # FIX ✅
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
    output = get_response(formatted_history, user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

with st.expander("Chat History", expanded=True):
    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            st.markdown(emoji.emojize(f":speech_balloon: **User {i}**: {st.session_state['past'][i]}"))
            st.markdown(emoji.emojize(f":robot: **Assistant {i}**: {st.session_state['generated'][i]}"))
