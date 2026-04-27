import streamlit as st
from dotenv import load_dotenv
import os

# Providers
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama

load_dotenv()

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="Multi-Provider Chatbot 🤖", layout="centered")
st.title("🤖 Multi-Provider AI Chatbot")

# ---------------- PROVIDERS & MODELS ----------------
providers = {
    "OpenAI": ["gpt-3.5-turbo", "gpt-4.1"],
    "Gemini": ["gemini-2.5-pro", "gemini-2.5-flash"],
    "Groq": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
    "Ollama": ["llama3.1", "gemma3"]
}

provider = st.selectbox("Select Provider", list(providers.keys()))
model = st.selectbox("Select Model", providers[provider])

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- DISPLAY CHAT ----------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- MODEL ROUTER ----------------
def get_llm(provider, model):
    if provider == "OpenAI":
        return ChatOpenAI(model=model, temperature=0)

    elif provider == "Gemini":
        return ChatGoogleGenerativeAI(model=model, temperature=0)

    elif provider == "Groq":
        return ChatGroq(model=model, temperature=0)

    elif provider == "Ollama":
        return ChatOllama(model=model)

# ---------------- CHAT INPUT ----------------
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Get model
    llm = get_llm(provider, model)

    # Prepare messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        *st.session_state.chat_history
    ]

    # Get response
    response = llm.invoke(messages)
    reply = response.content

    # Store assistant reply
    st.session_state.chat_history.append({"role": "assistant", "content": reply})

    # Display response
    with st.chat_message("assistant"):
        st.markdown(reply)