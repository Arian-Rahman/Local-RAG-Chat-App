import os
import streamlit as st
import sys
import types
import torch
from datetime import datetime
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# from streamlit.runtime.scriptrunner import rerun

from source.utils.paths import paths
from source.pdf_manager import process_and_save_pdf_pipeline
from source.vector_db_manager import vector_db_pipeline
from source.rag_chain_manager import build_question_answering_chain

# Prevent Streamlit from scanning torch.classes
if isinstance(torch.classes, types.ModuleType):
    torch.classes.__path__ = []  # Prevent __path__ errors
    
# --- Setup ---
st.set_page_config(page_title="🧠 RAG Assistant", layout="wide")
pdf_dir = paths["pdfs"]
os.makedirs(pdf_dir, exist_ok=True)

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append(("assistant", 
    "Hi! I’m your personal AI assistant. I’m here to help you understand and explore the PDFs you provide. Just ask me anything!",
    datetime.now()))
# --- Sidebar: PDF Controls ---

# --- Sidebar: PDF Controls ---
with st.sidebar:
    st.title("📁 Manage PDF")

    # Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    # Check if new file was uploaded and not processed yet
    if uploaded_file and ("last_uploaded_file" not in st.session_state or 
                            st.session_state.last_uploaded_file != uploaded_file.name):        # Remove any existing PDF
        for f in os.listdir(pdf_dir):
            if f.endswith(".pdf"):
                os.remove(os.path.join(pdf_dir, f))

        # Save new file
        new_file_path = os.path.join(pdf_dir, uploaded_file.name)
        with open(new_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.last_uploaded_file = uploaded_file.name  # Mark as processed

        st.success(f"Uploaded and replaced previous PDF with: {uploaded_file.name}")

        # Run the pipelines
        with st.spinner("Processing and indexing the new PDF..."):
            process_and_save_pdf_pipeline()
            vector_db_pipeline()
            logger.info("PDF processing and vector DB pipeline completed.")

        st.rerun()

    # Show currently available PDF
    current_pdfs = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    if current_pdfs:
        st.markdown(f"**Current PDF:** `{current_pdfs[0]}`")
    else:
        st.info("No PDF available.")
        


    st.markdown("---")
    st.title("⚙️ Settings")
    model_choice = st.selectbox("Choose a model:", [
        "llama3-chatqa",
        "deepseek-r1:14b"
    ])
    

# --- Main App ---
st.title("🧠 RAG Chat Assistant")
st.markdown("Ask a question based on your uploaded PDFs.")

# --- Scrollable Chat Box ---

from streamlit.components.v1 import html

if "chat_html" not in st.session_state:
    st.session_state.chat_html = """
<div style='height: 500px; overflow-y: auto; padding: 10px; border: 1px solid #ccc; border-radius: 10px; background-color: #f9f9f9;'>
"""

# --- Input Box and Buttons ---
with st.form(key="chat_form", clear_on_submit=True):
    user_question = st.text_input("Enter your question:")
    col1, col2 = st.columns([3, 1])
    with col1:
        submit = st.form_submit_button("Submit")
    with col2:
        clear = st.form_submit_button("Clear Chat")

    if submit and user_question.strip():
        st.session_state.chat_history.insert(0, ("user", user_question, datetime.now()))
        with st.spinner("Thinking..."):
            response = build_question_answering_chain(user_question, model_choice)
        st.session_state.chat_history.insert(0, ("assistant", response, datetime.now()))

    if clear:
        st.session_state.chat_history = []
        st.session_state.chat_history.append(("assistant", 
            "Hi! I’m your personal AI assistant. I’m here to help you understand and explore the PDFs you provide. Just ask me anything!",
                datetime.now()))


from markdown_it import MarkdownIt

md = MarkdownIt()

def update_chat_html():
    chat_html = """
    <div style='height: 500px; overflow-y: auto; padding: 10px; 
                border: 1px solid #ccc; border-radius: 10px; background-color: #f9f9f9;'>
    """
    for role, msg, timestamp in st.session_state.chat_history:
        align = "right" if role == "user" else "left"
        bg_color = "#DCF8C6" if role == "user" else "#F1F0F0"
        icon = "🧑‍💼" if role == "user" else "🤖"
        time_str = timestamp.strftime("%H:%M")

        # ✅ Convert markdown to HTML
        rendered_msg = md.render(msg)

        chat_html += f"""
        <div style='background-color:{bg_color}; padding:10px; margin:10px 0; 
                    border-radius:10px; float: {align}; clear: both; max-width: 80%; font-size: 18px;'>
            <b>{icon}</b><br>{rendered_msg}
            <div style='font-size: 10px; color: #888; text-align: {align};'>{time_str}</div>
        </div>
        """
    chat_html += "</div>"
    st.session_state.chat_html = chat_html


# ✅ After the form block, update the chat HTML
update_chat_html()
html(st.session_state.chat_html, height=400, scrolling=True)

