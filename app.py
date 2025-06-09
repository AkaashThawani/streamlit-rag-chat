import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
# --- THIS IS THE CORRECTED IMPORT BLOCK ---
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
# --- END OF CORRECTED IMPORTS ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage


load_dotenv()


def get_document_from_upload(uploaded_files):
    all_documents = []
    for uploaded_file in uploaded_files:
        tmpfile_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmpfile_path = tmp_file.name
            loader = PyPDFLoader(tmpfile_path)
            all_documents.extend(loader.load())
        finally:
            if tmpfile_path and os.path.exists(tmpfile_path):
                os.remove(tmpfile_path)

        return all_documents


def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004")
    return Chroma.from_documents(text_chunks, embeddings)


def get_context_retriver_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    retriever = vector_store.as_retriever()
    repharasing_prompt = ChatPromptTemplate.from_messages([
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    return create_history_aware_retriever(llm, retriever, repharasing_prompt)


def get_conversational_rag_chain(retriever):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    stuff_document_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the question based on the context provided \n\n {context}"),
        ("placeholder", "{chat_history}"),
        ("user", "{input}")
    ])
    
    stuff_document_chain = create_stuff_documents_chain(llm,stuff_document_prompt)
    return create_retrieval_chain(history_aware_retriever,stuff_document_chain)

def get_response(user_input):
    retriver_chain = get_context_retriver_chain(st.session_state.vector_store)
    conversational_rag_chain = get_conversational_rag_chain(retriver_chain)
    
    response = conversational_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

def main():
    st.set_page_config(page_title="Chat with your documents",page_icon=":books:")
    st.title("Chat with your documents :books:")
    st.write("Upload your PDF documents to start chatting with them.")
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(
                content="Hello! I am your virtual assistant. Please upload your PDF documents to start chatting with them."
            )
        ]
        
    with st.sidebar:
        st.subheader("Your Documents")
        uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        if st.button("Process"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    documents = get_document_from_upload(uploaded_files)
                    if documents:
                        text_chunks = get_text_chunks(documents)
                        st.session_state.vector_store = get_vector_store(text_chunks)
                        st.success("Documents processed successfully!")
                    else:
                        st.error("No valid documents found.")
            
            for message in st.session_state.chat_history:
                if isinstance(message, AIMessage):
                    with st.chat_message("assistant"):
                        st.write(message.content)
                elif isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.write(message.content)
            
            if user_query := st.chat_input("Ask a question about your documents"):
                st.session_state.chat_history.append(HumanMessage(content=user_query)) # type: ignore
                with st.chat_message("user"):
                    st.write(user_query)
                
                with st.spinner("Generating response..."):
                    response = get_response(user_query)
                    st.session_state.chat_history.append(AIMessage(content=response))
                    with st.chat_message("assistant"):
                        st.write(response) 

if __name__ == "__main__":
    main()
    