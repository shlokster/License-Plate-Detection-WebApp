import streamlit as st
from main import run_license_plate_recognition
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import re
pattern = r'\b(?:[A-Z]{1}[A-Z]{2}\d{2,3}[A-Z]{1,2}\d{4}|[A-Z]{2}\d{2,3}[A-Z]{1,2}\d{4})\b'
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

                        captured_plates = []
                        captured_time = []
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
                                    matches = re.search(pattern, text)
                                    if matches:
                                        trimmed_string = matches.group(0)
                                        trimmed_string = re.sub(r'^E(?=K)', '', trimmed_string)
                                    else:
                                        trimmed_string = None
                                    if trimmed_string != None:
                                        st.write(f"Detected License Plate Number at Frame {trimmed_string}: {frame_number}")
                                        captured_plates.append(trimmed_string)   
                                        captured_time.append(frame_number)

                            frame_number += 1

                            
                        # Release the video capture object and delete the temporary directory
                        cap.release()
                        st.warning("Frames extraction complete.")
                        df = pd.DataFrame({'License Plate Number': captured_plates, 'Time Stamps': captured_time})
                        df['Serial Number'] = range(1, len(df) + 1)

                        st.write(df)
                
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
