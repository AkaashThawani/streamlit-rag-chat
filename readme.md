# Chat with Your Documents: A RAG-powered Streamlit App

This is an interactive web application that allows you to upload one or more PDF documents and have a conversation with them. Using a Retrieval-Augmented Generation (RAG) architecture, this app leverages Google's Gemini Pro model through LangChain to understand and answer questions about the contents and metadata of your uploaded files.

![App Demo](placeholder.gif)
*(To create your own demo, use a tool like ScreenToGif or Kap to record the app in action and replace `placeholder.gif`)*

---

## ✅ Features

-   **Interactive Chat Interface:** A user-friendly, chat-based UI built with Streamlit.
-   **Multi-PDF Upload:** Upload one or several PDF documents to create a combined knowledge base.
-   **Metadata Aware:** The chatbot can answer questions about the document's metadata, such as filenames and page numbers (e.g., "In which file is 'Project Alpha' mentioned?").
-   **Content-Aware:** Ask complex questions about the text contained within your documents.
-   **Secure API Key Handling:** Uses a `.env` file to keep your Google API key secure and off of version control.
-   **Built with Modern Tools:** Leverages the power of LangChain, ChromaDB for in-memory vector storage, and the Google Gemini API.

---

## 🛠️ Technology Stack

-   **Frontend:** [Streamlit](https://streamlit.io/)
-   **Backend:** Python
-   **LLM:** Google Gemini Pro (`gemini-1.5-flash-latest`)
-   **Framework:** [LangChain](https://www.langchain.com/)
-   **Vector Store:** [ChromaDB](https://www.trychroma.com/) (in-memory)
-   **Embeddings:** Google `text-embedding-004`

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

-   Python 3.8 or higher
-   A Google Gemini API Key. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

### Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and Activate a Virtual Environment**
    This is highly recommended to keep project dependencies isolated.
    ```bash
    # Create the virtual environment
    python -m venv venv

    # Activate it (Windows)
    .\venv\Scripts\activate

    # Activate it (macOS/Linux)
    # source venv/bin/activate
    ```

3.  **Install Dependencies**
    With your virtual environment active, install all required packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Your Environment Variables**
    The application loads your API key from a `.env` file.
    -   Copy the example file to create your own local environment file:
        ```bash
        # On Windows
        copy .env.example .env

        # On macOS/Linux
        cp .env.example .env
        ```
    -   Open the new `.env` file in your code editor.
    -   Replace `YOUR_GEMINI_API_KEY_GOES_HERE` with your actual API key from Google AI Studio.

    > **IMPORTANT:** The `.env` file is listed in `.gitignore` and should **never** be committed to version control.

### Running the Application

1.  Make sure your virtual environment is activated (you should see `(venv)` in your terminal prompt).
2.  Run the following command from the project's root directory:
    ```bash
    streamlit run app.py
    ```
3.  Your default web browser will automatically open a new tab with the running application.

---

## ⚙️ How It Works

The application follows a standard RAG (Retrieval-Augmented Generation) pattern:

1.  **Document Processing (Ingestion):**
    -   When you upload PDF files and click "Process," the application uses `PyPDFLoader` to load the documents.
    -   It manually adds the **filename** and **page number** as metadata to each page.
    -   The documents are then split into smaller, overlapping text chunks using `RecursiveCharacterTextSplitter`.
    -   These chunks are converted into numerical representations (vector embeddings) using Google's embedding model.
    -   The chunks and their embeddings are stored in an in-memory ChromaDB vector store.

2.  **Chat Interaction (Retrieval & Generation):**
    -   When you ask a question, the application uses LangChain Expression Language (LCEL) to orchestrate a chain of events.
    -   **Retrieve:** Your question is embedded, and ChromaDB performs a similarity search to find the most relevant text chunks from the vector store.
    -   **Augment:** The retrieved chunks (including their metadata) are formatted and "stuffed" into a prompt along with your original question.
    -   **Generate:** This combined prompt is sent to the Gemini Pro model, which generates a final, conversational answer based *only* on the context it was given.

---

## 🔮 Future Improvements

-   **Add Chat History:** Implement a history-aware retriever to allow for conversational follow-up questions.
-   **Support More File Types:** Extend the app to handle `.docx`, `.txt`, and `.md` files using different LangChain document loaders.
-   **Persistent Vector Store:** Modify the ChromaDB setup to `persist_directory` so that you don't have to re-process documents every time you start the app.
-   **Streaming Responses:** Implement response streaming so the answer appears token-by-token, improving perceived performance.
-   **Deployment:** Deploy the application to a service like Streamlit Community Cloud or Hugging Face Spaces.