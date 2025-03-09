import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os
from langchain.memory import ConversationBufferMemory  # Updated import
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template, hide_st_style, footer
import pandas as pd


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_csv_text(csv_docs):
    """Process CSV files and convert to text format"""
    text = ""
    for csv in csv_docs:
        df = pd.read_csv(csv)
        # Add column names as context
        text += f"Dataset Columns: {', '.join(df.columns)}\n\n"
        # Convert each row to formatted text
        for idx, row in df.iterrows():
            text += f"Row {idx}:\n"
            for col, val in row.items():
                text += f"{col}: {val}\n"
            text += "\n"
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=2000, chunk_overlap=400, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # Updated model name
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.error("Please upload and process your data before starting the chat.")
        return

    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    load_dotenv()

    # Debug API key
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        st.error("Google API key not found. Please check your .env file.")
        return

    # Test API connection
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
        st.success("Successfully connected to Gemini API")
    except Exception as e:
        st.error(f"Error connecting to Gemini API: {str(e)}")
        return

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with AI with Custom Data ")
    user_question = st.text_input("Ask a question about your Data:")

    with st.sidebar:
        st.subheader("Your documents")
        file_type = st.radio("Choose file type:", ["PDF", "CSV"])

        if file_type == "PDF":
            uploaded_files = st.file_uploader(
                "Upload your PDFs here", accept_multiple_files=True, type=["pdf"]
            )
        else:
            uploaded_files = st.file_uploader(
                "Upload your CSVs here", accept_multiple_files=True, type=["csv"]
            )

        if st.button("Process"):
            if not uploaded_files:
                st.error(f"Please upload at least one {file_type} file.")
            else:
                with st.spinner("Processing"):
                    # Process based on file type
                    raw_text = (
                        get_pdf_text(uploaded_files)
                        if file_type == "PDF"
                        else get_csv_text(uploaded_files)
                    )
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success(f"Your {file_type} data has been processed successfully")

    if user_question:
        handle_userinput(user_question)

    st.markdown(hide_st_style, unsafe_allow_html=True)
    st.markdown(footer, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
