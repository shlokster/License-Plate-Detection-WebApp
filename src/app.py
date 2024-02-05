import streamlit as st
from main import run_license_plate_recognition
import os
import cv2
import re
pattern = r'\b(?:[A-Z]{1}[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}|[A-Z]{2}\d{2}[A-Z]{1,2}\d{4})\b'
frame_interval = 0.5

def app():
    st.header("License Plate Recognition Web App")
    st.subheader("Powered by YOLOv5")
    st.write("Welcome!")

    with st.form("my_uploader"):
        uploaded_file = st.file_uploader(
            "Upload file", type=["png", "jpg", "jpeg", "mp4"], accept_multiple_files=False
        )
        submit = st.form_submit_button(label="Upload")

        if uploaded_file is not None:
            # Create directory if it doesn't exist
            if not os.path.exists("temp"):
                os.makedirs("temp")

            # save uploaded file
            save_path = os.path.join("temp", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Check if the uploaded file is a video
            if uploaded_file.type.startswith('video'):
                # Process video here (e.g., display or analyze frames)
                st.video(uploaded_file)

                if submit and uploaded_file is not None:
                    # add spinner
                    with st.spinner(text="Detecting license plate ..."):
                        # display license plate as text

                        if not os.path.exists("frame_dir"):
                            os.makedirs("frame_dir")

                        cap = cv2.VideoCapture(save_path)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_interval_frames = int(fps * frame_interval)

                        frames = []
                        frame_number = 0
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            if frame_number % frame_interval_frames == 0:
                                # Append the frame to the list
                                frame_filename = os.path.join("frame_dir", f"frame_{frame_number + 1}.png")
                                cv2.imwrite(frame_filename, frame)

                                recognizer = run_license_plate_recognition(frame_filename)
                                text = recognizer.recognize_text()
                                print(text)
                                if text != None:
                                    matches = re.findall(pattern, text)
                                    if matches != []:
                                        st.write(f"Detected License Plate Number at Frame {frame_number}: {matches}")   

                            frame_number += 1

                            
                        # Release the video capture object and delete the temporary directory
                        cap.release()
                        st.warning("Frames extraction complete.")
                        st.balloons()
                        st.info("Temporary files are automatically cleaned up.")

                        # recognizer = run_video_split(temp_dir)
                        # text = recognizer.recognize_text_frames()
                        # st.write(f"Detected License Plate Number at : {text}")  
                
            else:
                # Process image here (e.g., display or analyze image)
                st.image(save_path)
                if submit and uploaded_file is not None:
                    # add spinner
                    with st.spinner(text="Detecting license plate ..."):
                        # display license plate as text
                        recognizer = run_license_plate_recognition(save_path)
                        text = recognizer.recognize_text()
                        print(text)
                        if text != None:
                            # matches = re.findall(pattern, text)
                            # if matches != []:
                                st.write(f"Detected License Plate Number: {text}")   


if __name__ == "__main__":
    app()
