import time
import streamlit as st
import requests

# Set the API key and base URL for NVIDIA Palmyra API
API_KEY = "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"
API_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

# Function to call the NVIDIA Palmyra API for financial analysis
def call_palmyra_api(question, retries=3, delay=5):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "writer/palmyra-fin-70b-32k",
        "messages": [
            {"role": "system", "content": "You are a financial assistant. Answer questions clearly and accurately based on general financial knowledge."},
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

# Streamlit app setup
st.title("General Financial Chatbot with NVIDIA Palmyra Model")
st.write("Ask any financial-related questions, and I will provide answers based on general financial knowledge.")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for chat in st.session_state.chat_history:
    st.write(f"**User**: {chat['question']}")
    st.write(f"**Bot**: {chat['answer']}")

# Input for user's question
question = st.text_input("Enter your question")

if st.button("Ask") and question:
    with st.spinner("Generating response..."):
        # Generate response from Palmyra API
        answer = call_palmyra_api(question)

        # Save to chat history
        st.session_state.chat_history.append({"question": question, "answer": answer})

        # Display the latest chat response
        st.write(f"**User**: {question}")
        st.write(f"**Bot**: {answer}")
