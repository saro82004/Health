
import json
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# ====== Load environment variables ======
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Make sure .env has: OPENAI_API_KEY=sk-xxxx

# ====== Load dataset ======
with open("Healthcare Dataset/health-data.json", "r", encoding="utf-8") as f:
    health_data = json.load(f)["diseases"]

# Convert dataset into LangChain Documents
docs = []
for disease in health_data:
    content = f"""
    Disease: {disease['name']}
    Description: {disease['description']}
    Symptoms: {', '.join(disease['symptoms'])}
    Treatments: {', '.join(disease['treatments'])}
    """
    docs.append(Document(page_content=content, metadata={"name": disease["name"]}))

# ====== Create vectorstore ======
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ====== Create QA chain ======
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, openai_api_key=OPENAI_API_KEY)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff"
)

# ====== Streamlit UI ======
st.set_page_config(page_title="Healthcare Chatbot", page_icon="🩺")
st.title("Healthcare Chatbot 🩺")
st.write("Ask me about diseases, symptoms, or treatments. I will only answer from my medical dataset.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Type your healthcare question..."):
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get dataset-based answer
    result = qa_chain({"query": prompt})
    raw_answer = result["result"]

    # Format answer with Markdown structure
    formatting_prompt = f"""
    Format the following healthcare answer in Markdown with:
    - **Bold headings** for sections
    - Bullet points for symptoms and treatments
    - A short description at the top
    - If possible, a table with columns: Symptoms, Treatments

    Answer:
    {raw_answer}
    """
    formatted_answer = llm.predict(formatting_prompt)

    # Show assistant message
    with st.chat_message("assistant"):
        st.markdown(formatted_answer)
    st.session_state.messages.append({"role": "assistant", "content": formatted_answer})
