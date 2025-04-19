import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Get the API key from Streamlit secrets
api_key = st.secrets["GOOGLE_API_KEY"]

# Configure the API key
genai.configure(api_key=api_key)

st.set_page_config(page_title="DocBot", page_icon=":robot_face:")
st.title("🤖Mini DocBot📝")
st.write(
    "This is a RAG based mini DocBot that can answer questions based on the content of a PDF document. "
    "You can upload a PDF file, and the bot will extract the text and provide answers to your questions."
)
st.caption("Powered by Gemini✨")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am a medical chatbot. How can I help you today?"}]

