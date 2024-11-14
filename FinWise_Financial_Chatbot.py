import time
import pdfplumber
import streamlit as st
import requests
from sentence_transformers import SentenceTransformer, util
import torch

# Set the API key and base URL for NVIDIA Palmyra API
API_KEY = "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"
API_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

# Set the project name
PROJECT_NAME = "FinWise Chatbot"

# Streamlit app setup
st.set_page_config(page_title=PROJECT_NAME, layout="wide")

# Load the sentence transformer model for semantic similarity checking
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

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

# Function to call the NVIDIA Palmyra API
def call_palmyra_api(messages, retries=3, delay=5):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "writer/palmyra-fin-70b-32k",
        "messages": messages,
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 1024,
        "stream": False
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_ENDPOINT, headers=headers, json=data)

            if response.status_code == 200:
                try:
                    json_response = response.json()
                    return json_response.get("choices", [{}])[0].get("message", {}).get("content", "No response received from the API.")
                except ValueError:
                    return f"Unexpected response format: {response.text}"
            elif response.status_code == 502:
                print(f"502 Bad Gateway Error - Retry {attempt + 1}/{retries}")
                time.sleep(delay)
            else:
                if response.status_code == 404:
                    return "Error: 404 Not Found - Please check if the API endpoint URL is correct."
                elif response.status_code == 401:
                    return "Error: 401 Unauthorized - Please verify your API key."
                else:
                    return f"Error: {response.status_code}, {response.text}"

        except requests.exceptions.RequestException as e:
            return f"Request failed: {e}"

    return "Failed to get a successful response after multiple retries."

# Function to summarize chunks of text using the model
def summarize_text(chunk):
    messages = [
        {"role": "system", "content": "You are an assistant who summarizes text in a concise manner."},
        {"role": "user", "content": f"Please summarize the following text:\n{chunk}"}
    ]
    return call_palmyra_api(messages)

# Function to chunk text into smaller parts
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Initialize session state variables
if 'pdf_embeddings' not in st.session_state:
    st.session_state['pdf_embeddings'] = None
if 'pdf_chunks' not in st.session_state:
    st.session_state['pdf_chunks'] = []
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Main title and description
st.title(f"📄 {PROJECT_NAME}: Your Financial Document Assistant")

# Write a paragraph explaining the methodology and tech stack used
st.write("""
Welcome to **FinWise Chatbot**, an advanced financial document assistant that leverages state-of-the-art natural language processing and retrieval techniques to provide accurate and context-aware responses to your queries.

This application employs a combination of **Retrieval-Augmented Generation (RAG)** and cutting-edge language models to process and understand financial documents:

- **Methodology**: The system extracts textual data from uploaded PDFs using `pdfplumber`, then segments and summarizes the content through interaction with NVIDIA's **Palmyra language model**, specifically fine-tuned for financial contexts. The summarized data is embedded using **Sentence Transformers** to create high-dimensional vector representations. When a query is posed, the application computes semantic similarity between the query and document embeddings via cosine similarity metrics using **PyTorch**, retrieving the most contextually relevant information. This context is then supplied to the Palmyra model to generate precise and informative answers, effectively integrating document-specific knowledge with the model's extensive financial expertise.

- **Tech Stack**: The application is built using **Python** and **Streamlit** for the web interface, leveraging libraries such as `pdfplumber` for PDF text extraction, `sentence-transformers` for embedding generation, and `torch` for tensor computations. API communication is facilitated through `requests`, interfacing with NVIDIA's **Palmyra API** to harness the capabilities of large-scale language models in the financial domain.

Please upload a financial PDF document to begin interacting with FinWise Chatbot.
""")

# Add a horizontal line to separate sections
st.markdown("---")

# Create two columns: left for 'Chat History', right for 'Upload Document and Ask Questions'
col_left, col_right = st.columns([2.5, 1.5])

with col_right:
    st.header("Upload Document and Ask Questions")
    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # If a PDF file is uploaded
    if uploaded_file is not None:
        with st.spinner("Extracting and processing text from PDF..."):
            # Extract text from PDF
            extracted_text = extract_text_from_pdf(uploaded_file)

        if extracted_text:
            # Chunk and summarize text
            with st.spinner("Summarizing the document and creating embeddings..."):
                chunks = chunk_text(extracted_text)
                summarized_chunks = []
                for i, chunk in enumerate(chunks):
                    summary = summarize_text(chunk)
                    summarized_chunks.append(summary)

                # Create embeddings for the summarized chunks
                chunk_embeddings = embedding_model.encode(summarized_chunks, convert_to_tensor=True)

                # Store the embeddings and chunks in session state
                st.session_state['pdf_embeddings'] = chunk_embeddings
                st.session_state['pdf_chunks'] = summarized_chunks

            st.success("Document processed successfully. Now you can ask questions based on the document or general finance.")

    # Input for user's question and "Ask" button aligned to the right
    question = st.text_input("Enter your question")

    # Create two columns within the right column to align the "Ask" button to the right
    btn_col1, btn_col2 = st.columns([3, 1])

    with btn_col2:
        ask_button = st.button("Ask")

    if ask_button:
        if question:
            # Embed the user's question
            question_embedding = embedding_model.encode(question, convert_to_tensor=True)

            # If PDF data is available, find relevant chunks
            context = ""
            if st.session_state['pdf_embeddings'] is not None and st.session_state['pdf_embeddings'].numel() > 0:
                # Compute cosine similarities
                cosine_scores = util.cos_sim(question_embedding, st.session_state['pdf_embeddings'])[0]

                # Get the top k most relevant chunks
                top_k = min(3, len(cosine_scores))
                top_results = torch.topk(cosine_scores, k=top_k)

                relevant_chunks = []
                for score, idx in zip(top_results.values, top_results.indices):
                    if score > 0.5:  # Adjust the threshold as needed
                        relevant_chunks.append(st.session_state['pdf_chunks'][idx])

                if relevant_chunks:
                    context = "\n".join(relevant_chunks)

            # Prepare messages for the API
            system_prompt = "You are a helpful financial assistant. Use the provided context if it's relevant to answer the user's question. If the context is not helpful, rely on your general financial knowledge to answer the question."

            messages = [
                {"role": "system", "content": system_prompt}
            ]

            if context:
                messages.append({"role": "user", "content": f"Context:\n{context}"})

            messages.append({"role": "user", "content": question})

            # Generate response from Palmyra API
            with st.spinner("Generating response..."):
                answer = call_palmyra_api(messages)

            # Save to chat history
            st.session_state.chat_history.append({"question": question, "answer": answer})

        else:
            st.warning("Please enter a question.")

with col_left:
    st.header("Chat History")
    # Display chat history
    for chat in st.session_state.chat_history:
        st.markdown(f"**User**: {chat['question']}")
        st.markdown(f"**FinWise**: {chat['answer']}")
        st.markdown("---")
