# Mini-DocBot

## Overview
A lightweight Retrieval-Augmented Generation (RAG) chatbot for document-based question answering. Mini-DocBot processes PDF documents, converts content into vector embeddings using FAISS, and generates context-aware responses using Google's Gemini 1.5 Flash model.

## Features
- Document-based Q&A through RAG architecture
- PDF processing and intelligent knowledge retrieval
- Interactive interface via Streamlit

## Demo
Try it live: [mini-docbot.streamlit.app](https://mini-docbot.streamlit.app/)

## Requirements
- streamlit - Web application framework
- google-generativeai - Google Gemini API access
- langchain & langchain_google_genai - LLM framework integration
- langchain-community - Additional LangChain components
- PyPDF2 - PDF document processing
- chromadb - Vector database for document storage
- faiss-cpu - Vector similarity search library
- streamlit secrets - Environment variable management