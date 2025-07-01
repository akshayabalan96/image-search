import streamlit as st
import requests
import os

st.title("🔍 Image Similarity Search")
st.markdown("Upload a query image and select a folder to find similar images")

query_image = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png"])
folder_path = st.text_input("Enter path to local folder with images")

if st.button("Search"):
    if query_image is None or not folder_path:
        st.warning("Please upload an image and enter a folder path")
    else:
        # Save uploaded image temporarily
        with open("temp_query.jpg", "wb") as f:
            f.write(query_image.getbuffer())

        # Call backend API
        with open("temp_query.jpg", "rb") as img:
            files = {
                "query_image": img,
                "folder_path": (None, folder_path)
            }
            response = requests.post("http://127.0.0.1:8000/find-similar/", files=files)

        if response.status_code == 200:
            results = response.json()["results"]
            st.success("Top similar images")
            for res in results:
                st.write(f"{res['filename']} - Similarity: {res['similarity']:.4f}")
        else:
            st.error("Something went wrong. Check server logs.")

        os.remove("temp_query.jpg")
