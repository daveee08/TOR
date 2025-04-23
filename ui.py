import streamlit as st
import requests
from datetime import datetime

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Transcript Extractor", layout="centered")
st.title("📄 Transcript Extractor")

uploaded_file = st.file_uploader("Upload a PDF or Excel file", type=["pdf", "xls", "xlsx"])

if uploaded_file:
    st.info("Uploading and extracting...")
    with st.spinner("Processing..."):

        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        response = requests.post(f"{API_URL}/upload", files=files)

        if response.status_code == 200:
            data = response.json()
            st.success(f"Extracted content from: {data['filename']}")

            for section in data["extracted_content"]:
                with st.expander(f"Section {section['id']}"):
                    if section.get("text"):
                        st.markdown(f"**Page {section.get('page', '-')}:**")
                        st.text(section["text"])
                    elif section.get("columns"):
                        st.json(section["columns"])
                    else:
                        st.warning("No content extracted in this section.")

            if st.button("💾 Save to Database"):
                save_response = requests.post(f"{API_URL}/save", json=data)
                if save_response.status_code == 200:
                    st.success("Data saved successfully.")
                else:
                    st.error("Failed to save data.")
        else:
            st.error(f"Upload failed: {response.text}")
