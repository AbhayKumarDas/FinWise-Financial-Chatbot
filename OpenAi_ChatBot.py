import openai
import pdfplumber
import numpy as np
import pandas as pd
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re
import matplotlib.pyplot as plt

# Set up your OpenAI API key securely from environment variable
import os
os.environ["OPENAI_API_KEY"] = "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"

api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    openai.api_key = api_key
else:
    st.warning("OpenAI API key not set. Please check your environment variables.")

# Flag for enabling/disabling API calls (useful for testing)
enable_api_calls = os.getenv('ENABLE_API_CALLS', 'false').lower() == 'true'

# Initialize OpenAI embeddings for document chunking
embeddings = OpenAIEmbeddings()

# Helper function to extract and structure financial data from PDF
def extract_text_from_pdf(pdf_file):
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")
        return ""

# Chunk text using LangChain’s Text Splitter for efficient embedding-based retrieval
def chunk_text(text, chunk_size=1000, overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to extract and calculate financial metrics
def extract_financial_data(text):
    try:
        revenue = float(re.search(r"Revenue:?\s?\$?(\d+[,\.]?\d*)", text).group(1).replace(",", ""))
        expenses = float(re.search(r"Expenses:?\s?\$?(\d+[,\.]?\d*)", text).group(1).replace(",", ""))
        assets = float(re.search(r"Assets:?\s?\$?(\d+[,\.]?\d*)", text).group(1).replace(",", ""))
        liabilities = float(re.search(r"Liabilities:?\s?\$?(\d+[,\.]?\d*)", text).group(1).replace(",", ""))
        
        # Calculate more detailed metrics
        net_income = revenue - expenses
        debt_to_equity_ratio = liabilities / (assets - liabilities) if (assets - liabilities) != 0 else float('inf')
        profit_margin = net_income / revenue if revenue != 0 else 0
        current_ratio = assets / liabilities if liabilities != 0 else float('inf')
        return_on_assets = net_income / assets if assets != 0 else 0

        data = {
            "revenue": revenue,
            "expenses": expenses,
            "assets": assets,
            "liabilities": liabilities,
            "net_income": net_income,
            "debt_to_equity_ratio": debt_to_equity_ratio,
            "profit_margin": profit_margin,
            "current_ratio": current_ratio,
            "return_on_assets": return_on_assets
        }

        return data
    except AttributeError:
        st.warning("Could not extract financial data accurately. Please ensure the PDF has the correct information.")
        return None

# Create embeddings and store them in FAISS for document search
def create_faiss_index(chunks):
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# Query FAISS index for relevant context
def query_index(query, vector_store, k=3):
    docs = vector_store.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in docs])
    return context

# Use OpenAI API to get an answer with relevant context
def ask_openai(question, context):
    if enable_api_calls:
        print("API call is enabled. Proceeding to make an OpenAI API call.")
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Context: {context}\n\nQuestion: {question}\nAnswer:",
            max_tokens=200,
            temperature=0.2
        )
        return response.choices[0].text.strip()
    else:
        print("API call is disabled. Returning placeholder answer.")
        return "API call disabled for testing purposes."

# Function to plot financial metrics
def plot_financial_metrics(financial_metrics):
    labels = list(financial_metrics.keys())
    values = list(financial_metrics.values())

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# Streamlit setup
st.title("Advanced Financial Assistant Chatbot")
st.write("Upload your financial document (PDF), and ask finance-related questions.")

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner('Extracting text from PDF...'):
        text = extract_text_from_pdf(uploaded_file)
    
    if text:
        st.success("PDF text extracted successfully.")
        
        # Calculate and display financial metrics
        financial_metrics = extract_financial_data(text)
        if financial_metrics:
            st.write("### Calculated Financial Metrics:")
            st.json(financial_metrics)

            # Plot the financial metrics
            st.write("### Financial Metrics Overview:")
            plot_financial_metrics(financial_metrics)
        
        # Chunk text for embedding-based retrieval
        chunks = chunk_text(text)

        # Create a FAISS index from the chunks
        vector_store = create_faiss_index(chunks)

        # Chat interface
        st.write("### Ask a question about the financial document")
        question = st.text_input("Your Question")
        
        if question:
            with st.spinner('Fetching response...'):
                # Retrieve relevant chunks as context
                context = query_index(question, vector_store)
                
                # Include financial metrics in context for better answers
                full_context = f"{context}\n\nFinancial Metrics: {financial_metrics}"
                
                # Get answer from OpenAI API or return message if disabled
                answer = ask_openai(question, full_context)
                
                # Display the answer
                st.write("#### Answer:")
                st.write(answer)
