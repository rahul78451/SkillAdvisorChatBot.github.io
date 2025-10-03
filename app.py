import asyncio
import threading
import time 
import os

# Fix for: RuntimeError: There is no current event loop in thread 'ScriptRunner.scriptThread'
if threading.current_thread() is threading.main_thread():
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
else:
    asyncio.set_event_loop(asyncio.new_event_loop())

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SerpAPIWrapper

import streamlit as st
import emoji
from itertools import zip_longest


# ================== API KEYS ==================
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Google API key not found in secrets.toml. Please add it to .streamlit/secrets.toml")
    st.stop() 

if "SERPAPI_API_KEY" not in st.secrets:
    st.error("SerpAPI key not found in secrets.toml. Please add it to .streamlit/secrets.toml")
    st.stop() 

google_api_key = st.secrets["GOOGLE_API_KEY"]
serpapi_api_key = st.secrets["SERPAPI_API_KEY"]

genai.configure(api_key=google_api_key)


# ================== APP TITLE ==================
st.title(f"Career Advisor Chatbot {emoji.emojize(':robot:')}")

# Directory & FAISS DB Path
pdf_dir = 'pdf'
VECTOR_DB_PATH = "career_advisor_faiss_index" 


# ================== DATABASE LOADING ==================
if "vectors" not in st.session_state:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )

    if os.path.exists(VECTOR_DB_PATH):
        # Always load from disk (pre-created DB)
        with st.spinner("Loading Database from disk..."):
            st.session_state["vectors"] = FAISS.load_local(
                VECTOR_DB_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        st.success("✅ Database loaded successfully from disk!")
    else:
        st.error("❌ FAISS Database not found. Please run the embedding script locally to create it.")
        st.stop()


# ================== RESPONSE FUNCTION ==================
def get_response(history,user_message,temperature=0):
    DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and a Career Advisor. 
    The Advisor guides the user regarding jobs, interests and other domain selection decisions.
    It follows the previous conversation to do so.

    Relevant pieces of previous conversation:
    {context},

    Useful information from career guidance books:
    {text}, 

    Useful information about career guidance from Web:
    {web_knowledge},

    Current conversation:
    Human: {input}
    Career Expert:"""

    PROMPT = PromptTemplate(
        input_variables=['context','input','text','web_knowledge'], 
        template=DEFAULT_TEMPLATE
    )
    docs = st.session_state["vectors"].similarity_search(user_message) 

    params = {
        "engine": "bing",
        "gl": "us",
        "hl": "en",
    }
    search = SerpAPIWrapper(params=params, serpapi_api_key=serpapi_api_key)
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


# ================== HISTORY UTILS ==================
def get_history(history_list):
    history = ''
    for message in history_list:
        if message['role']=='user':
            history = history+'input '+message['content']+'\n'
        elif message['role']=='assistant':
            history = history+'output '+message['content']+'\n'
    return history


# ================== STREAMLIT UI ==================
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

    output = get_response(formatted_history,user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)


with st.expander("Chat History", expanded=True):
    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])): 
            st.markdown(emoji.emojize(f":speech_balloon: **User {str(i)}**: {st.session_state['past'][i]}"))
            st.markdown(emoji.emojize(f":robot: **Assistant {str(i)}**: {st.session_state['generated'][i]}"))
