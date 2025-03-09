import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template, hide_st_style, footer
import pandas as pd
from typing import List, Union
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate

# Update this deprecated import
from langchain_community.callbacks.manager import get_openai_callback


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
        # Convert each row to formatted text without row numbers
        for _, row in df.iterrows():  # Using _ instead of idx
            text += "Record:\n"  # Changed from "Row {idx}" to "Record"
            for col, val in row.items():
                text += f"{col}: {val}\n"
            text += "\n"
    return text


# 1. Update Text Chunking for Better Context
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,  # Reduced for more granular chunks
        chunk_overlap=200,  # Adjusted overlap for better context
        length_function=len,
    )

    # Enhanced text cleaning
    cleaned_text = (
        text.replace("  ", " ").replace("\n\n", "\n").replace("\t", " ").strip()
    )

    chunks = text_splitter.split_text(cleaned_text)

    # Add more context to chunks
    chunks_with_metadata = [
        {
            "content": chunk,
            "chunk_id": i,
            "position": (
                "start" if i == 0 else "end" if i == len(chunks) - 1 else "middle"
            ),
        }
        for i, chunk in enumerate(chunks)
    ]
    return chunks_with_metadata


# 2. Optimize Vector Store Configuration
def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        task_type="retrieval_query",
        dimension=768,  # Specify embedding dimension
    )

    vectorstore = FAISS.from_documents(
        documents=[
            Document(
                page_content=chunk["content"],
                metadata={"chunk_id": chunk["chunk_id"], "position": chunk["position"]},
            )
            for chunk in text_chunks
        ],
        embedding=embeddings,
    )

    # Optimize FAISS parameters
    vectorstore.index.nprobe = 4  # Increased search probes
    return vectorstore


def save_vectorstore(vectorstore, file_type: str) -> None:
    """Save the vectorstore to disk"""
    try:
        save_path = f"vectorstore_{file_type.lower()}"
        vectorstore.save_local(save_path)
        st.session_state.vectorstore_path = save_path
        return True
    except Exception as e:
        st.error(f"Error saving vectorstore: {str(e)}")
        return False


def load_vectorstore(file_type: str):
    """Load the vectorstore from disk with safe deserialization"""
    try:
        save_path = f"vectorstore_{file_type.lower()}"
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            task_type="retrieval_query",
        )
        if os.path.exists(save_path):
            # Add allow_dangerous_deserialization flag for trusted local files
            vectorstore = FAISS.load_local(
                save_path,
                embeddings,
                allow_dangerous_deserialization=True,  # Only for trusted local files
            )
            return vectorstore
        return None
    except Exception as e:
        st.error(f"Error loading vectorstore: {str(e)}")
        return None


def get_conversation_chain(vectorstore):
    # Initialize the LLM with optimized parameters
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,  # Reduced for more focused responses
        max_output_tokens=2048,  # Increased token limit
        top_p=0.80,  # Adjusted for better response quality
        top_k=20,  # Reduced for more focused token selection
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    # Enhanced prompt template for better responses
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a precise document analysis assistant. Follow these rules strictly:
            1. Always analyze the provided context thoroughly
            2. Give direct, clear answers based only on the context
            3. If multiple relevant pieces of information exist, combine them coherently
            4. If the answer isn't in the context, say "I don't have enough information to answer that question"
            5. Never make assumptions or provide information not present in the context
            6. Sunway college is located behind Maitidevi temple, Maitidevi, Kathmandu.
            """,
            ),
            (
                "assistant",
                "I will analyze the documents and provide accurate information based solely on the given context.",
            ),
            ("human", "Here is the relevant context:\n{context}"),
            ("human", "Using only the above context, answer this question: {question}"),
        ]
    )

    # Improved memory configuration
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        input_key="question",
    )

    # Enhanced retrieval configuration
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={
                "k": 4,  # Increased relevant chunks
                "score_threshold": 0.7,  # Higher threshold for better matches
                "fetch_k": 6,  # Fetch more candidates
                "search_type": "similarity",  # Changed to similarity search
            }
        ),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": prompt_template,
            "document_separator": "\n\n",  # Clear separation between documents
        },
        return_source_documents=True,
        verbose=True,  # Enable for debugging
    )
    return conversation_chain


# 4. Modified Response Handler
def handle_userinput(user_question: str) -> None:
    try:
        if st.session_state.conversation is None:
            st.error("Please upload and process your data before starting the chat.")
            return

        with st.spinner("Analyzing documents..."):
            try:
                # Process the question
                response = st.session_state.conversation(
                    {
                        "question": user_question,
                        "chat_history": st.session_state.chat_history or [],
                    }
                )

                if "answer" in response:
                    answer = response["answer"].strip()

                    # Handle empty or invalid responses
                    if not answer:
                        st.error(
                            "I couldn't generate a valid response. Please try rephrasing your question."
                        )
                        return

                    # Update chat history
                    if st.session_state.chat_history is None:
                        st.session_state.chat_history = []

                    # Add interaction to history
                    st.session_state.chat_history.extend(
                        [
                            {"role": "user", "content": user_question},
                            {"role": "assistant", "content": answer},
                        ]
                    )

                    # Display interaction
                    st.write(
                        user_template.replace("{{MSG}}", user_question),
                        unsafe_allow_html=True,
                    )
                    st.write(
                        bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True
                    )

                else:
                    st.error(
                        "No valid response received. Please try asking a different question."
                    )

            except Exception as e:
                st.error(f"Error processing response: {str(e)}")
                st.error("Please try rephrasing your question.")

    except Exception as e:
        st.error(f"System error: {str(e)}")
        st.error("Please try again or reload the application.")


def merge_vectorstores(existing_store, new_store):
    """Merge new vectorstore with existing one"""
    try:
        # Merge the indices
        existing_store.merge_from(new_store)
        return existing_store
    except Exception as e:
        st.error(f"Error merging vectorstores: {str(e)}")
        return None


def main():
    load_dotenv()

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False
    if "vectorstore_path" not in st.session_state:
        st.session_state.vectorstore_path = None

    st.write(css, unsafe_allow_html=True)

    # Page layout
    st.title("Document Chat Assistant")
    st.markdown(
        """
    Upload your documents and start asking questions about their content.
    Supports PDF and CSV files.
    """
    )

    # Sidebar for file upload
    with st.sidebar:
        st.subheader("Document Upload")
        file_type = st.radio("Choose file type:", ["PDF", "CSV"])

        # Try to load existing vectorstore
        existing_vectorstore = load_vectorstore(file_type)
        if existing_vectorstore is not None:
            st.success(f"Loaded existing {file_type} vectorstore!")
            st.session_state.conversation = get_conversation_chain(existing_vectorstore)
            st.session_state.file_processed = True

        uploaded_files = st.file_uploader(
            f"Upload your {file_type} files",
            accept_multiple_files=True,
            type=[file_type.lower()],
            help=f"Upload one or more {file_type} files to analyze",
        )

        if st.button("Process Documents", type="primary", key="process_button"):
            if not uploaded_files:
                st.error(f"Please upload at least one {file_type} file.")
            else:
                with st.spinner("Processing documents..."):
                    try:
                        # Process new documents
                        raw_text = (
                            get_pdf_text(uploaded_files)
                            if file_type == "PDF"
                            else get_csv_text(uploaded_files)
                        )
                        text_chunks = get_text_chunks(raw_text)
                        new_vectorstore = get_vectorstore(text_chunks)

                        # Load existing vectorstore if available
                        existing_vectorstore = load_vectorstore(file_type)

                        if existing_vectorstore is not None:
                            # Merge new data with existing
                            st.info("Merging with existing data...")
                            final_vectorstore = merge_vectorstores(
                                existing_vectorstore, new_vectorstore
                            )
                            if final_vectorstore is None:
                                st.error("Failed to merge vectorstores")
                                return
                        else:
                            final_vectorstore = new_vectorstore

                        # Save updated vectorstore
                        if save_vectorstore(final_vectorstore, file_type):
                            st.success("Vectorstore updated successfully!")

                        st.session_state.conversation = get_conversation_chain(
                            final_vectorstore
                        )
                        st.session_state.file_processed = True
                        st.success("Documents processed and merged successfully!")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")

        # Add clear cache button
        if st.button("Clear Saved Data", type="secondary"):
            try:
                if st.session_state.vectorstore_path and os.path.exists(
                    st.session_state.vectorstore_path
                ):
                    import shutil

                    shutil.rmtree(st.session_state.vectorstore_path)
                st.session_state.conversation = None
                st.session_state.chat_history = None
                st.session_state.file_processed = False
                st.session_state.vectorstore_path = None
                st.success("Saved data cleared successfully!")
            except Exception as e:
                st.error(f"Error clearing data: {str(e)}")

    # Main chat interface
    if st.session_state.file_processed:
        user_question = st.text_input(
            "Ask a question about your documents:",
            placeholder="What would you like to know about the uploaded documents?",
            key="user_input",
        )

        if user_question:
            handle_userinput(user_question)
    else:
        st.info("Please upload and process your documents to start chatting")

    st.markdown(hide_st_style, unsafe_allow_html=True)
    st.markdown(footer, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
