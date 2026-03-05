import streamlit as st
import numpy as np
import ollama
from pdf_reader import extract_text_from_pdf
from embeddings import create_embeddings
from rag_pipeline import create_faiss_index, search
from sentence_transformers import SentenceTransformer

# -------------------------
# Embedding model
# -------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(page_title="AI Study Assistant", page_icon="📚", layout="centered")
st.title("📚 AI Study Assistant")
st.write("Ask questions directly from your handwritten notes using AI.")

# -------------------------
# Initialize Session State
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "index" not in st.session_state:
    st.session_state.index = None

if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = None

# -------------------------
# Upload PDF
# -------------------------
st.header("Upload Your Handwritten Notes")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    help="Upload your handwritten or typed notes in PDF format."
)

if uploaded_file:
    with st.spinner("Processing PDF..."):
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Extract text and create embeddings
        text_chunks = extract_text_from_pdf("uploaded_file.pdf")
        embeddings = create_embeddings(text_chunks)
        index = create_faiss_index(embeddings)

        # Save to session
        st.session_state.index = index
        st.session_state.text_chunks = text_chunks

    st.success("✅ Notes indexed successfully! You can now ask questions.")

# -------------------------
# Ask Question
# -------------------------
if st.session_state.index is not None:
    st.header("Ask a Question")
    question = st.text_input("Type your question here:")

    if question:
        with st.spinner("Generating answer..."):
            # Get query embedding
            query_embedding = embedding_model.encode([question])
            distances, indices = search(st.session_state.index, query_embedding)

            # Retrieve relevant text chunks
            retrieved_text = ""
            pages = []
            for i in indices[0]:
                retrieved_text += st.session_state.text_chunks[i][1] + "\n"
                pages.append(st.session_state.text_chunks[i][0])

            # -------------------------
            # Improved Confidence Score
            # -------------------------
            weights = 1 / (np.array(distances[0]) + 1e-5)  # higher weight for closer chunks
            confidence = np.average(100 - np.array(distances[0])*100, weights=weights)
            confidence = max(confidence, 50)  # ensure minimum 50%
            confidence = round(confidence, 2)

            # Prepare prompt
            prompt = f"""
Use ONLY the notes below to answer.

Notes:
{retrieved_text}

Question:
{question}

If answer not found say:
"I don't know based on the notes."
"""

            # Get answer from Ollama
            response = ollama.chat(
                model="phi3:mini",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response["message"]["content"]

        # Display Answer
        st.subheader("Answer")
        st.write(answer)

        # Display Confidence
        st.subheader("Confidence Score")
        st.write(f"{confidence}%")

        # Display Source Pages
        st.subheader("Source Pages")
        st.write(sorted(list(set(pages))))

        # Save chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "confidence": confidence
        })

# -------------------------
# Display Chat History
# -------------------------
if st.session_state.chat_history:
    st.header("Chat History")
    for chat in reversed(st.session_state.chat_history):
        with st.container():
            st.markdown("**Question:**")
            st.write(chat["question"])
            st.markdown("**Answer:**")
            st.write(chat["answer"])
            st.markdown("**Confidence:**")
            st.write(f"{chat['confidence']}%")
            st.markdown("---")
