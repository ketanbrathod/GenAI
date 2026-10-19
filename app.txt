import streamlit as st
import requests

st.title("My LLM Application")

file = st.file_uploader("Upload your PDF document", type=["pdf"])
question = st.text_input("Ask me anything")

if file and question:
    if st.button("Get Answer"):
        response = requests.post(
            "http://127.0.0.1:8000/ask",
            files={"file": (file.name, file.getvalue(), "application/pdf")},
            data={"question": question}
        )
        st.write(response.json()["answer"])