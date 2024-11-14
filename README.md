# FinWise Chatbot: An Advanced Financial Document Assistant

## Overview

**FinWise Chatbot** is an intelligent financial document assistant that leverages cutting-edge artificial intelligence (AI) and machine learning (ML) techniques to provide accurate, context-aware responses to user queries. By integrating **Retrieval-Augmented Generation (RAG)** with specialized language models, FinWise offers an interactive platform for users to engage with their financial documents and obtain precise information.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Methodology](#methodology)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Integration](#api-integration)
- [Performance Considerations](#performance-considerations)
- [Security and Compliance](#security-and-compliance)
- [License](#license)

## Features

- **Document Upload**: Accepts financial documents in PDF format for processing.
- **Text Extraction**: Utilizes advanced parsing techniques to extract text from PDFs.
- **Summarization**: Employs transformer-based language models for concise document summarization.
- **Semantic Embeddings**: Generates high-dimensional vector representations using pre-trained models.
- **Contextual Question Answering**: Answers queries by retrieving relevant document context and leveraging a specialized language model.
- **Interactive Chat Interface**: Provides a user-friendly chat interface with persistent conversation history.
- **Scalability**: Designed to handle large documents and multiple concurrent user queries efficiently.

## Architecture

The FinWise Chatbot architecture comprises several interconnected components:

1. **Frontend Interface**:
   - Built with **Streamlit**, offering an interactive UI for file upload, question input, and displaying responses.

2. **Document Processing Pipeline**:
   - **Text Extraction**: Parses PDF files using `pdfplumber`.
   - **Text Chunking**: Splits extracted text into manageable chunks for processing.
   - **Summarization Module**: Summarizes text chunks via the **NVIDIA Palmyra language model API**.

3. **Embedding Generation**:
   - Utilizes `SentenceTransformer` models to generate semantic embeddings of summarized text.
   - Embeddings are stored for efficient similarity computations.

4. **Retrieval Mechanism**:
   - Calculates cosine similarity between user query embeddings and document embeddings.
   - Retrieves top-k relevant document chunks based on similarity scores.

5. **Response Generation**:
   - Constructs prompts incorporating retrieved context.
   - Generates answers using the **NVIDIA Palmyra language model API**.

6. **State Management**:
   - Maintains session states for embeddings, document chunks, and chat history using `st.session_state`.

## Methodology

**FinWise Chatbot** employs advanced AI and ML techniques to deliver accurate and context-rich responses:

- **Retrieval-Augmented Generation (RAG)**:
  - Combines information retrieval with generative models to produce context-aware answers.
  - Enhances the model's knowledge base with document-specific information.

- **Transformer Models**:
  - Uses transformer architectures for both embedding generation (`SentenceTransformer`) and language modeling (NVIDIA Palmyra).
  - Benefits from attention mechanisms to capture contextual relationships.

- **Semantic Similarity**:
  - Computes high-dimensional cosine similarity to identify relevant document sections.
  - Leverages vector space models for efficient similarity computations.

- **Natural Language Understanding (NLU)**:
  - Parses and interprets user queries to generate accurate answers.
  - Handles complex financial terminology and concepts.

- **Session Management**:
  - Utilizes Streamlit's session state to maintain context across user interactions.
  - Ensures a seamless conversational experience.

### Detailed Workflow

1. **Text Extraction and Summarization**:
   - **Extraction**: Uses `pdfplumber` to parse text from uploaded PDFs.
   - **Chunking**: Splits the text into chunks (e.g., 500 words) to manage processing limitations.
   - **Summarization**: Summarizes each chunk using the NVIDIA Palmyra API, reducing verbosity and focusing on key information.

2. **Embedding Generation**:
   - **Embeddings**: Converts summarized text chunks into embeddings using a pre-trained `SentenceTransformer` model (`all-MiniLM-L6-v2`).
   - **Storage**: Stores embeddings for quick retrieval during query processing.

3. **User Query Processing**:
   - **Embedding**: Generates an embedding for the user's query.
   - **Similarity Computation**: Calculates cosine similarity between the query embedding and document embeddings.

4. **Context Retrieval**:
   - **Ranking**: Selects top-k relevant chunks based on similarity scores and a predefined threshold.
   - **Context Assembly**: Aggregates relevant chunks to form the context for response generation.

5. **Response Generation**:
   - **Prompt Construction**: Prepares a prompt for the language model, including system role, context, and user query.
   - **API Interaction**: Sends the prompt to the NVIDIA Palmyra API to generate the response.
   - **Display**: Presents the generated answer to the user in the chat interface.

## Tech Stack

- **Programming Language**: Python 3.7+
- **Frontend**: Streamlit (for interactive web interface)
- **Libraries and Frameworks**:
  - `pdfplumber`: For PDF text extraction.
  - `sentence-transformers`: For generating semantic embeddings.
  - `torch` (PyTorch): Underlying tensor computations for embeddings.
  - `requests`: For HTTP requests to the NVIDIA Palmyra API.
- **APIs**:
  - **NVIDIA Palmyra Language Model API**: Specialized in financial contexts, used for summarization and response generation.

## Installation

### Prerequisites

- **Python**: Version 3.7 or higher.
- **NVIDIA Palmyra API Key**: Obtain from NVIDIA's developer portal.

### Install Dependencies

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/yourusername/finwise-chatbot.git
cd finwise-chatbot
```

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```txt
openai
langchain
langchain-community
pdfplumber
numpy
pandas
streamlit
matplotlib
python-dotenv
sentence-transformers 
streamlit>=1.22.0
pdfplumber>=0.10.0
sentence-transformers>=2.2.2
torch>=1.9.0
requests>=2.25.1 
```

## Usage - FinWise_Financial_Chatbot.py //This is the main python file to be run

1. **Configure API Key**:

   - Replace `"your_api_key_here"` in the `app.py` script with your actual NVIDIA Palmyra API key.

2. **Run the Application**:

   ```bash
   python -m streamlit run FinWise_Financial_Chatbot.py
   ```

3. **Interact with FinWise Chatbot**:

   - **Upload a Document**: Use the file uploader on the right to select a financial PDF document.
   - **Ask Questions**: Input your query in the text field and click "Ask".
   - **View Responses**: The chatbot's answers will appear in the "Chat History" on the left.

## API Integration

**FinWise Chatbot** integrates with the **NVIDIA Palmyra API** for advanced language modeling capabilities:

- **Summarization**:

  - **Input**: Text chunks from the document.
  - **Process**: Sends a prompt to the API requesting a concise summary.
  - **Output**: Summarized text used for embedding generation.

- **Response Generation**:

  - **Input**: Constructed prompt including system role, context, and user query.
  - **Process**: API generates a response based on provided context and query.
  - **Output**: Context-aware answer displayed to the user.

### API Call Structure

- **Endpoint**: `https://integrate.api.nvidia.com/v1/chat/completions`
- **Headers**:
  - `Authorization`: Bearer token with the API key.
  - `Content-Type`: `application/json`
- **Payload**:

  ```json
  {
    "model": "writer/palmyra-fin-70b-32k",
    "messages": [...],
    "temperature": 0.2,
    "top_p": 0.7,
    "max_tokens": 1024,
    "stream": false
  }
  ```

- **Error Handling**: Implements retry mechanisms and handles HTTP errors gracefully.

## Performance Considerations

- **Computational Efficiency**:

  - **Batch Processing**: For large documents, consider processing text chunks in batches to optimize API calls and reduce latency.
  - **Parallelization**: Utilize multiprocessing or asynchronous calls where appropriate to improve performance.

- **Resource Management**:

  - **Memory Usage**: Be mindful of memory consumption when handling large embeddings. Implement garbage collection or memory profiling if necessary.
  - **Caching**: Employ caching strategies for repeated queries to reduce redundant computations.

- **Scalability**:

  - **Load Balancing**: In a production environment, distribute the workload across multiple instances to handle high traffic.
  - **Containerization**: Use Docker or similar technologies for deployment consistency and scalability.

## Security and Compliance

- **API Key Management**:

  - Store the API key securely, preferably using environment variables or a configuration file excluded from version control.

- **Data Privacy**:

  - Ensure compliance with data protection regulations (e.g., GDPR, CCPA) when handling sensitive financial documents.
  - Implement encryption for data at rest and in transit if necessary.

- **User Authentication**:

  - For multi-user deployments, implement authentication mechanisms to protect user data and session states.

---

**Disclaimer**: **FinWise Chatbot** is intended for informational purposes only. The accuracy of responses is dependent on the quality of the uploaded documents and the capabilities of the underlying language models. Users should verify critical information independently.
