import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Model names
EMBEDDING_MODEL = "models/embedding-001"
CHAT_MODEL = "gemini-2.0-flash"  # Or "gemini-pro", "gemini-2.0-flash", or "gemini-2.0-flash-lite"

def get_pdf_text(pdf_docs):
    """Extracts text from multiple PDF documents."""
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        return ""

def get_text_chunks(text):
    """Splits text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Creates and saves a FAISS vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        if not os.path.exists("faiss_index"):
            os.makedirs("faiss_index")
        vector_store.save_local("faiss_index")
        st.success("FAISS index created successfully!")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_direct_model():
    """Create a direct chat model without document retrieval."""
    model = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.7)
    return model

def get_conversation_chain(vector_store=None):
    """Creates a conversational chain with or without document retrieval."""
    model = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.7)
    
    if vector_store is None:
        # Direct chat without documents
        return model
    else:
        # Chat with document retrieval
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            output_key='answer'
        )
        return chain

def handle_user_input(user_question):
    """Handles user input and retrieves answers."""
    # Add question to chat history
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    # Display updated chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if st.session_state.vector_store is not None and st.session_state.mode == "document_qa":
                    # Document Q&A mode
                    chain = get_conversation_chain(st.session_state.vector_store)
                    response = chain({"question": user_question})
                    answer = response['answer']
                else:
                    # General chat mode
                    model = get_direct_model()
                    response = model.invoke(user_question)
                    answer = response.content
                
                st.write(answer)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"Error generating response: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

def init_session_state():
    """Initialize session state variables."""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your AI assistant. Ask me any coding questions or general queries, or upload PDFs to chat with your documents."}
        ]
    
    if 'mode' not in st.session_state:
        st.session_state.mode = "general_chat"

def main():
    """Main function for the Streamlit app."""
    st.set_page_config(page_title="AI Assistant", layout="wide")
    init_session_state()
    
    st.header("AI Coding and General Knowledge Assistant", divider='rainbow')
    
    # Sidebar for settings and document upload
    with st.sidebar:
        st.title("Settings")
        
        mode = st.radio(
            "Chat Mode", 
            ["General Chat", "Document Q&A"],
            index=0 if st.session_state.mode == "general_chat" else 1
        )
        
        st.session_state.mode = "general_chat" if mode == "General Chat" else "document_qa"
        
        if st.session_state.mode == "document_qa":
            st.subheader("Document Upload")
            pdf_docs = st.file_uploader(
                "Upload PDF Files for Document Q&A", 
                accept_multiple_files=True
            )
            if st.button("Process Documents"):
                if pdf_docs:
                    with st.spinner("Processing documents..."):
                        raw_text = get_pdf_text(pdf_docs)
                        if raw_text:
                            text_chunks = get_text_chunks(raw_text)
                            st.session_state.vector_store = get_vector_store(text_chunks)
                            st.success("Documents processed successfully! You can now ask questions about them.")
                else:
                    st.warning("Please upload PDF files.")
        
        st.subheader("Options")
        if st.button("Clear Chat History"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat history cleared. How can I help you?"}
            ]
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # User input
    user_question = st.chat_input("Ask me anything...")
    if user_question:
        handle_user_input(user_question)

if __name__ == "__main__":
    main()
