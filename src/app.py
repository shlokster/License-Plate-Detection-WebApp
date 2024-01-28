import streamlit as st
from main import run_license_plate_recognition
import os


def app():
    st.header("License Plate Recognition Web App")
    st.subheader("Powered by YOLOv5")
    st.write("Welcome!")

    with st.form("my_uploader"):
        uploaded_file = st.file_uploader(
            "Upload image", type=["png", "jpg", "jpeg"], accept_multiple_files=False
        )
        submit = st.form_submit_button(label="Upload")

    if uploaded_file is not None:
        # Create directory if it doesn't exist
        if not os.path.exists("temp"):
            os.makedirs("temp")
            
        # save uploaded image
        save_path = os.path.join("temp", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    if submit and uploaded_file is not None:
        # add spinner
        with st.spinner(text="Detecting license plate ..."):
            # display license plate as text
            recognizer = run_license_plate_recognition(save_path)
            text = recognizer.recognize_text()
            st.write(f"Detected License Plate Number: {text}")
            
            # show uploaded image with bounding box
            st.image(save_path, caption="Uploaded Image", use_column_width=True)


if __name__ == "__main__":
    app()
