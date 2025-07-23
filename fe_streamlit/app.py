import streamlit as st
import requests
import re

API_URL = "http://localhost:8000"

def highlight_text(text, phrase):
    if not phrase:
        return text
    # Escape for regex, ignore case
    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    return pattern.sub(lambda m: f'<mark>{m.group(0)}</mark>', text)

st.title("Document Management FE (Streamlit)")

st.header("1. Insert File")
with st.form("insert_form"):
    file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
    submitted = st.form_submit_button("Insert")
    if submitted and file:
        files = {"file": (file.name, file, file.type)}
        resp = requests.post(f"{API_URL}/insert", files=files)
        if resp.ok:
            file_id = resp.json().get("file_id")
            st.success(f"Inserted! file_id: {file_id}")
        else:
            st.error(f"Error: {resp.text}")

st.header("2. Search")
with st.form("search_form"):
    query = st.text_input("Enter your search query")
    search_submitted = st.form_submit_button("Search")
    if search_submitted and query:
        resp = requests.post(f"{API_URL}/search", data={"query": query})
        if resp.ok:
            results = resp.json().get("results", [])
            if results:
                for r in results:
                    st.write(f"**File:** {r['file_id']} | **Pos:** {r['position']} | **Score:** {r['combined_score']:.3f}")
                    st.markdown(highlight_text(r['text'], query), unsafe_allow_html=True)
                    st.markdown("---")
            else:
                st.info("No results found.")
        else:
            st.error(f"Error: {resp.text}")

st.header("3. Delete File")
with st.form("delete_form"):
    del_file_id = st.text_input("File ID to delete")
    del_submitted = st.form_submit_button("Delete")
    if del_submitted and del_file_id:
        resp = requests.post(f"{API_URL}/delete", data={"file_id": del_file_id})
        if resp.ok:
            st.success(f"Deleted file_id: {del_file_id}")
        else:
            st.error(f"Error: {resp.text}") 