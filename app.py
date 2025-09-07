import asyncio
import threading

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

import os
from itertools import zip_longest

# Check for the API keys using Streamlit's secrets management
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Google API key not found in secrets.toml. Please add it to .streamlit/secrets.toml")
    st.stop() # Stops the script if the key is missing

if "SERPAPI_API_KEY" not in st.secrets:
    st.error("SerpAPI key not found in secrets.toml. Please add it to .streamlit/secrets.toml")
    st.stop() # Stops the script if the key is missing


# Retrieve the keys from Streamlit secrets
google_api_key = st.secrets["GOOGLE_API_KEY"]
serpapi_api_key = st.secrets["SERPAPI_API_KEY"]

# Configure genai with the key from secrets
genai.configure(api_key=google_api_key)


st.title(f"Career Advisor Chatbot {emoji.emojize(':robot:')}")

# Define your directory containing PDF files here
pdf_dir = 'pdf'

if "pdf_texts" not in st.session_state:
    temp_pdf_texts = []
    with st.spinner("Creating a Database..."):
        try:
            for file in os.listdir(pdf_dir):
                if file.endswith('.pdf'):
                    loader = PyPDFLoader(os.path.join(pdf_dir, file))
                    documents = loader.load()
                    text = " ".join([doc.page_content for doc in documents])
                    temp_pdf_texts.append(text)
        except FileNotFoundError:
            st.error(f"Error: The directory '{pdf_dir}' was not found. Please make sure it exists and contains your PDF files.")
            st.stop()
        
        st.session_state["pdf_texts"] = temp_pdf_texts
        pdf_list = list(st.session_state["pdf_texts"])
        pdfDatabase = " ".join(pdf_list)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(pdfDatabase)
        
        # Explicitly pass the API key from secrets
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )

        if "vectors" not in st.session_state: 
            vectors = FAISS.from_texts(chunks, embeddings)
            st.session_state["vectors"] = vectors
    st.success("Database creation completed!")

def get_response(history,user_message,temperature=0):

    DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an Career Advisor. The Advisor guides the user regaring jobs,interests and other domain selection decsions.
    It follows the previous conversation to do so

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
        input_variables=['context','input','text','web_knowledge'], template=DEFAULT_TEMPLATE
    )
    docs = st.session_state["vectors"].similarity_search(user_message)


    params = {
    "engine": "bing",
    "gl": "us",
    "hl": "en",
    }

    # Pass the SerpAPI key from secrets to the wrapper
    search = SerpAPIWrapper(params=params, serpapi_api_key=serpapi_api_key)

    web_knowledge=search.run(user_message)


    # **FIXED**: Using the updated, stable model name
    gemini_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro", # Use a stable model name
        temperature=temperature,
        google_api_key=google_api_key
    )

    conversation_with_summary = LLMChain(
        llm=gemini_model,
        prompt=PROMPT,
        verbose=False
    )
    response = conversation_with_summary.predict(context=history,input=user_message,web_knowledge=web_knowledge,text = docs)
    return response

# Function to get conversation history
def get_history(history_list):
    history = ''
    for message in history_list:
        if message['role']=='user':
            history = history+'input '+message['content']+'\n'
        elif message['role']=='assistant':
            history = history+'output '+message['content']+'\n'
    
    return history


# Streamlit UI
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
