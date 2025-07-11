import streamlit as st
import os
from src.models.rag.rag_model import RAGModel

@st.cache_resource
def load_rag_model():
    return RAGModel.load("dataset/models/rag/")

rag = load_rag_model()

def rag_inference(query):
    if not query or query.strip() == "":
        return "Please Write your Question"
    
    try:
        results = rag.predict([query])
        if isinstance(results, tuple):
            result = results[0][0]['answer']
        elif isinstance(results, list):
            result = results[0] if results else "No result"
        else:
            result = str(results)
        return result
    except Exception as e:
        return f"Erreur : {str(e)}"

#Streamlit Interface
st.set_page_config(page_title=" AI Ticketing Assistant", layout="wide")
st.title("🤖 AI Ticketing Assistant")

col1, col2 = st.columns(2)

with col1:
    user_input = st.text_area("Your Question", placeholder="Describe your problem")

with col2:
    if user_input:
        with st.spinner("Is Generating..."):
            response = rag_inference(user_input)
        st.text_area("Assistant Answer", value=response, height=300, disabled=True)