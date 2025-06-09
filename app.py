import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from operator import itemgetter

# --- Load Environment Variables ---
load_dotenv()

# --- Core Functions ---

def process_and_store_documents(uploaded_files):
    """
    Processes uploaded PDF files, adds metadata, and stores them in a vector store.
    """
    all_chunks = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            tmpfile_path = tmpfile.name

        loader = PyPDFLoader(tmpfile_path)
        pages = loader.load_and_split()
        
        # Add metadata to each page
        for page in pages:
            page.metadata["source"] = uploaded_file.name
        
        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(pages)
        all_chunks.extend(chunks)
        
        os.remove(tmpfile_path)

    # Create the vector store from all chunks
    api_key = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key) # type: ignore
    vector_store = Chroma.from_documents(documents=all_chunks, embedding=embeddings)
    
    return vector_store

def get_rag_chain(vector_store):
    """
    Builds a RAG (Retrieval-Augmented Generation) chain.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    retriever = vector_store.as_retriever()

    # The prompt template for the RAG chain
    prompt = ChatPromptTemplate.from_template(
        """You are an expert assistant. Answer the user's question based on the following context.
Be sure to use the metadata (like 'source' and 'page') when it is relevant to the question.

Context:
{context}

Question:
{question}

Answer:"""
    )

    def format_docs(docs):
        # Formats the retrieved documents to be passed to the LLM
        return "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'N/A')}\n"
            f"Content: {doc.page_content}"
            for doc in docs
        )

    # The RAG chain using LangChain Expression Language (LCEL)
    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- Streamlit Application ---

def main():
    st.set_page_config(page_title="Chat with Your Docs", page_icon="📄")
    st.title("📄 Chat with Your Documents")
    
    # Initialize session state variables
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello! Upload your PDFs in the sidebar to get started.")]
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    # --- Sidebar for File Uploading ---
    with st.sidebar:
        st.subheader("Your Documents")
        uploaded_files = st.file_uploader(
            "Upload PDFs and click 'Process'", accept_multiple_files=True, type="pdf"
        )
        if st.button("Process Documents",type="primary",use_container_width=True,disabled=not uploaded_files):
            if not uploaded_files:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    # Process the files and create the RAG chain
                    vector_store = process_and_store_documents(uploaded_files)
                    st.session_state.rag_chain = get_rag_chain(vector_store)
                    
                    # Store the names of processed files
                    st.session_state.processed_files = [f.name for f in uploaded_files]
                    st.success("Documents processed successfully!")

    # --- Main Chat Area ---
    if st.session_state.processed_files:
        st.info(f"Currently chatting with: {', '.join(st.session_state.processed_files)}")

    # Display chat history
    for msg in st.session_state.chat_history:
        if isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)

    # Handle user input
    if user_query := st.chat_input("Ask a question about your documents...",disabled=not st.session_state.processed_files):
        if st.session_state.rag_chain is None:
            st.warning("Please process your documents first.")
        else:
            st.session_state.chat_history.append(HumanMessage(content=user_query)) # type: ignore
            st.chat_message("user").write(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Invoke the RAG chain with a dictionary input
                    response = st.session_state.rag_chain.invoke({"question": user_query})
                    st.write(response)
                    st.session_state.chat_history.append(AIMessage(content=response))

if __name__ == "__main__":
    main()