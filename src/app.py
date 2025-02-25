import streamlit as st
from pdf_processor.processor import extract_text_from_pdf
from chatbot.query import query_llm

st.title("PDF Query Chatbot")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Text", pdf_text, height=300)

    query = st.text_input("Enter your query")
    if st.button("Submit"):
        response = query_llm(pdf_text, query)
        st.write(response)