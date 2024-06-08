import streamlit as st
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from pydantic.v1 import SecretStr
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import re

st.title("Document-GPT")

uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)

# Set OpenAI API key from Streamlit secrets
api_key = SecretStr("...")

# Initialize the ChatOpenAI client
client = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=api_key,
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Text cleaning
def clean_text(text):
    text = text.replace("\n", " ")  # replace newline characters with space
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = text.strip()  # remove leading and trailing whitespace
    return text


ai_msg = None
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        _, ext = os.path.splitext(file_name)
        if ext == ".pdf":
            # Read the file into a byte stream
            pdf_bytes = uploaded_file.read()
            # Convert the byte stream into images
            images = convert_from_bytes(pdf_bytes)
            # Iterate over the images and extract text
            extracted_text = ""
            for i in range(len(images)):
                text = pytesseract.image_to_string(images[i], lang="eng")
                extracted_text += text
            # Clean the extracted text
            cleaned_text = clean_text(extracted_text)
            if cleaned_text:  # Check if the extracted text is not empty
                # Prepare messages for the chat
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that identifies and extracts entities like names, dates, addresses, emails, phone numbers, etc. Extracting relationships between entities. Summarizing key information from the document.",
                    },
                    {
                        "role": "assistant",
                        "content": cleaned_text,
                    },
                ]
                # Invoke the chat model
                ai_msg = client.invoke(messages)
            else:
                st.error("Extracted text was empty.")
        else:
            st.error(f"{file_name} is not a PDF file.")

## Prompting
user_query = st.chat_input("Ask Questions?")
if user_query:
    with st.chat_message("user"):
        st.write(user_query)
    if ai_msg:  # Ensure ai_msg is not None
        # Prepare the input for the rag_chain
        input_for_rag_chain = {
            "input": user_query,
            "context": ai_msg.content,
        }

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(context=input_for_rag_chain["context"]),
            },
            {
                "role": "user",
                "content": user_query,
            },
        ]
        response = client.invoke(messages)
        total_tokens = response.response_metadata["token_usage"]["total_tokens"]
        with st.chat_message("assistant"):
            st.write(response.content)
            st.write(total_tokens)
    else:
        st.error("No context available to answer the question.")
