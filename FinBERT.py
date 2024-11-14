import pdfplumber
import streamlit as st
import faiss
import torch
from transformers import BertForQuestionAnswering, BertTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import os

# Set the correct path for your pre-trained model files
MODEL_PATH = r"C:\Users\ABHAY KUMAR DAS\OneDrive\Documents\Codes\FinanceChatbot\FinBERT-QA\model\trained\finbert-qa"

# Load the FinBERT QA model using PyTorch from a local directory
@st.cache_resource
def load_qa_pipeline():
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForQuestionAnswering.from_pretrained(MODEL_PATH)
    except OSError as e:
        st.error(f"Error loading model/tokenizer: {e}. Please check the model path and required files.")
        return None
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, framework="pt", device=0 if torch.cuda.is_available() else -1)
    return qa_pipeline

# Load a sentence transformer for embedding-based retrieval
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')  # A lightweight, efficient model for embedding

# Load models
qa_pipeline = load_qa_pipeline()
embedding_model = load_embedding_model()

# Function to extract text from the PDF
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

# Function to split the text into smaller chunks for retrieval
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Function to create FAISS index with embeddings of the text chunks
def create_faiss_index(chunks):
    embeddings = [embedding_model.encode(chunk) for chunk in chunks]
    embedding_size = len(embeddings[0])
    index = faiss.IndexFlatL2(embedding_size)
    index.add(torch.tensor(embeddings).numpy())
    return index, embeddings

# Function to retrieve the most relevant chunks using FAISS
def retrieve_relevant_chunks(question, chunks, index, embeddings, top_k=3):
    question_embedding = embedding_model.encode(question)
    _, indices = index.search(torch.tensor([question_embedding]).numpy(), top_k)
    return [chunks[idx] for idx in indices[0]]

# Function to generate an answer using the FinBERT QA model
def answer_question(question, context):
    input_data = {
        "question": question,
        "context": context
    }
    result = qa_pipeline(input_data)
    return result['answer']

# Streamlit app setup
st.title("RAG-Based Financial PDF Chatbot")
st.write("Upload a financial document (PDF) and ask any finance-related questions.")

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# If a PDF file is uploaded
if uploaded_file is not None:
    with st.spinner("Extracting and processing text from PDF..."):
        # Extract and chunk text from PDF
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)

        # Create FAISS index for retrieval
        index, embeddings = create_faiss_index(chunks)

    if text:
        st.success("PDF text extracted and indexed successfully.")
        
        # Display extracted text (optional)
        st.write("### Extracted Text (First 3000 characters):")
        st.write(text[:3000] + "..." if len(text) > 3000 else text)
        
        # Allow user to ask questions
        st.write("### Ask a question about the document")
        question = st.text_input("Enter your question")
        
        if question:
            with st.spinner("Retrieving relevant information and generating answer..."):
                # Retrieve relevant chunks
                relevant_chunks = retrieve_relevant_chunks(question, chunks, index, embeddings, top_k=3)
                combined_context = " ".join(relevant_chunks)  # Combine top relevant chunks
                
                # Generate answer using FinBERT-QA model
                answer = answer_question(question, combined_context)
                
                # Display the answer
                st.write("#### Answer:")
                st.write(answer)
