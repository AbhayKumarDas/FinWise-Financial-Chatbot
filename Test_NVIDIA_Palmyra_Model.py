import time
import pdfplumber
import streamlit as st
import requests
from sentence_transformers import SentenceTransformer, util

# Set the API key and base URL for NVIDIA Palmyra API
API_KEY = "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"
API_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

# Load a sentence transformer model for semantic similarity checking
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

# Function to call the NVIDIA Palmyra API for financial analysis
def call_palmyra_api(context, question, retries=3, delay=5):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "writer/palmyra-fin-70b-32k",
        "messages": [
            {"role": "system", "content": "You are a financial assistant. Use the provided document context if available. If the user's question is unrelated, provide general financial information."},
            {"role": "user", "content": context},
            {"role": "user", "content": question}
        ],
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 1024,
        "stream": False
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_ENDPOINT, headers=headers, json=data)

            # Debugging: Print response details
            print("Response Status Code:", response.status_code)
            print("Response Headers:", response.headers)
            print("Response Text:", response.text)

            if response.status_code == 200:
                try:
                    json_response = response.json()
                    return json_response.get("choices", [{}])[0].get("message", {}).get("content", "No response received from the API.")
                except ValueError:
                    return f"Unexpected response format: {response.text}"
            elif response.status_code == 502:
                # If a 502 error occurs, log it and retry after a delay
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

# Function to determine if the question is related to the context from the PDF
def is_question_related(context, question, threshold=0.6):
    context_embedding = embedding_model.encode(context, convert_to_tensor=True)
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(context_embedding, question_embedding).item()
    return similarity_score >= threshold

# Streamlit app setup
st.title("Financial Document Chatbot with NVIDIA Palmyra Model")
st.write("Upload a financial document (PDF) to interact with it through questions, or ask any general financial questions.")

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# If a PDF file is uploaded
if uploaded_file is not None:
    with st.spinner("Extracting and processing text from PDF..."):
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(uploaded_file)

    if extracted_text:
        st.success("PDF text extracted successfully. Now you can ask me questions based on the document or general finance.")

        # Store the extracted text in session state for future reference
        st.session_state['context'] = extracted_text

# Chatbot Interface
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for chat in st.session_state.chat_history:
    st.write(f"**User**: {chat['question']}")
    st.write(f"**Bot**: {chat['answer']}")

# Input for user's question
question = st.text_input("Enter your question")

if st.button("Ask") and question:
    # Determine if there is context from the PDF available
    context = ""
    if 'context' in st.session_state:
        # Determine if the question is related to the PDF content
        with st.spinner("Determining the relevance of the question..."):
            related = is_question_related(st.session_state['context'], question)
        
        if related:
            context = st.session_state['context']
        else:
            context = "No specific document context. Provide general financial information."
    else:
        context = "No specific document context. Provide general financial information."

    # Generate response from Palmyra API
    with st.spinner("Generating response..."):
        answer = call_palmyra_api(context, question)

    # Save to chat history
    st.session_state.chat_history.append({"question": question, "answer": answer})

    # Display the latest chat response
    st.write(f"**User**: {question}")
    st.write(f"**Bot**: {answer}")

