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

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Upload a PDF document, and I'll be ready to answer your questions about it!"}]
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# Create and save FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    save_path = os.path.expanduser("./faiss_index")  # writable path
    vector_store.save_local(save_path)

# Load conversational AI chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "answer is not available in the context"; don't provide a wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Handling user input and generate responses
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# PDF upload section below the description
st.subheader("Upload Documents")
uploaded_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and not st.session_state.pdf_processed:
    with st.spinner("Processing PDFs..."):
        # Extract text from PDFs
        raw_text = get_pdf_text(uploaded_files)
        
        # Get text chunks
        text_chunks = get_text_chunks(raw_text)
        
        # Create and save vector store
        get_vector_store(text_chunks)
        
        st.session_state.pdf_processed = True
        st.success("PDF processed successfully!")
        st.session_state.messages = [{"role": "assistant", "content": "I've processed your documents! Ask me anything about them."}]
        st.rerun()

# Main chat interface
st.subheader("Chat with your documents")

# Display all messages in the chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle new user input
if user_question := st.chat_input():
    if not st.session_state.pdf_processed:
        st.warning("Please upload a PDF document first.")
    else:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.chat_message("user").write(user_question)
        
        with st.spinner("Thinking..."):
            # Generate the response
            response = user_input(user_question)
            
            # Add assistant message to session state
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
