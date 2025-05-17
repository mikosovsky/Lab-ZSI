import streamlit as st
import os
import tempfile
import fitz  # PyMuPDF
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.schema import Document

# Konfiguracja OpenRouter
os.environ["OPENAI_API_KEY"] = st.secrets["API_KEY"]
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

st.set_page_config(page_title="📚 Chatbot z Plików", layout="wide")

# Funkcja do ekstrakcji tekstu z PDF przez PyMuPDF
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    texts = []
    for page in doc:
        texts.append(page.get_text())
    return "\n".join(texts)

# Funkcja do stworzenia bazy wektorowej z dokumentów
def create_vectorstore(file_list):
    docs = []
    for file in file_list:
        suffix = file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        if suffix == "pdf":
            text = extract_text_from_pdf(tmp_path)
            docs.append(Document(page_content=text, metadata={"name": file.name}))
        elif suffix == "txt":
            loader = TextLoader(tmp_path)
            docs.extend(loader.load())
    
    # Podział tekstu na fragmenty
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    # Embedding (tani model z HuggingFace)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # FAISS indexing
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

# Lewy pasek: uploader
st.sidebar.header("📎 Prześlij pliki")
uploaded_files = st.sidebar.file_uploader("PDF lub TXT", type=["pdf", "txt"], accept_multiple_files=True)

# Główna kolumna: czat
st.header("🤖 Chatbot z opcjonalnym kontekstem z plików")

# Inicjalizacja FAISS tylko jeśli pliki istnieją
vectorstore = None
if uploaded_files:
    with st.spinner("🔍 Tworzę bazę wiedzy z plików..."):
        vectorstore = create_vectorstore(uploaded_files)
        st.sidebar.success("📚 Pliki załadowane i zindeksowane!")

# Model OpenRouter (darmowy)
llm = ChatOpenAI(model_name="mistralai/mistral-7b-instruct", temperature=0.3)

# Inicjalizacja czatu
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Wprowadzenie użytkownika
user_input = st.chat_input("Zadaj pytanie...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.spinner("🧠 Myślę..."):

        # Jeśli mamy wektory, użyj RAG, w przeciwnym razie tylko LLM
        if vectorstore:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            answer = qa.run(user_input)
        else:
            answer = llm.predict(user_input)

        st.session_state.chat_history.append(("bot", answer))

# Wyświetl historię rozmowy
for sender, message in st.session_state.chat_history:
    with st.chat_message("user" if sender == "user" else "assistant"):
        st.markdown(message)
