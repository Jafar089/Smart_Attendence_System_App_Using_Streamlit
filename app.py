import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import streamlit as st
import pandas as pd
import zipfile
from io import BytesIO

# Check dlib version
import dlib

st.write(f"dlib version: {dlib.__version__}")

# Streamlit app setup
st.title("Face Recognition Attendance System")

# Directory to store uploaded images
path = 'uploaded_images'
if not os.path.exists(path):
    os.makedirs(path)

def save_uploaded_zip(uploaded_zip):
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(path)
    return st.success("Images extracted successfully")

def load_images_from_directory(directory):
    images = []
    person_names = []
    my_list = os.listdir(directory)
    for cu_img in my_list:
        current_img = cv2.imread(os.path.join(directory, cu_img))
        if current_img is not None:
            images.append(current_img)
            person_names.append(os.path.splitext(cu_img)[0])
    return images, person_names

def face_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            encode = encodings[0]
            encode_list.append(encode)
    return encode_list

def attendance(name):
    file_name = 'Attendance.csv'
    if not os.path.isfile(file_name):
        with open(file_name, 'w') as f:
            f.write('Name,Time,Date\n')

    with open(file_name, 'r+') as f:
        my_data_list = f.readlines()
        name_list = [line.split(',')[0] for line in my_data_list]
        if name not in name_list:
            time_now = datetime.now()
            t_str = time_now.strftime('%H:%M:%S')
            d_str = time_now.strftime('%d/%m/%Y')
            f.writelines(f'{name},{t_str},{d_str}\n')
            st.write(f"Recorded attendance for {name}")

# Upload zip file section
st.header("Upload Zip File of Images")
uploaded_zip = st.file_uploader("Choose a zip file", type="zip")
if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(path)
    st.success("Uploaded and extracted images successfully")

# Load images and encodings
images, person_names = load_images_from_directory(path)
encode_list_known = face_encodings(images)
st.write('All Encodings Complete!!!')

# Button to start/stop the webcam
if 'run' not in st.session_state:
    st.session_state.run = False

def start():
    st.session_state.run = True
    st.session_state.cap = cv2.VideoCapture(0)

def stop():
    st.session_state.run = False
    if 'cap' in st.session_state and st.session_state.cap.isOpened():
        st.session_state.cap.release()
        cv2.destroyAllWindows()
    # Display the Attendance CSV file and provide a download link
    if os.path.isfile('Attendance.csv'):
        df = pd.read_csv('Attendance.csv')
        st.write("Attendance Record")
        st.dataframe(df)

        # Provide a download link
        st.download_button(
            label="Download Attendance CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='Attendance.csv',
            mime='text/csv'
        )

# Button to empty the attendance file
def clear_attendance():
    open('Attendance.csv', 'w').close()
    st.success("Attendance file cleared")

clear_button = st.button("Clear Attendance File", on_click=clear_attendance)

start_button = st.button("Start Camera", on_click=start)
stop_button = st.button("Stop Camera", on_click=stop)

frame_window = st.image([])

while st.session_state.run:
    ret, frame = st.session_state.cap.read()
    if not ret:
        st.error("Failed to capture image")
        break

    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    faces_current_frame = face_recognition.face_locations(faces)
    encodes_current_frame = face_recognition.face_encodings(faces, faces_current_frame)

    current_names = set()

    for encode_face, face_loc in zip(encodes_current_frame, faces_current_frame):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_dis = face_recognition.face_distance(encode_list_known, encode_face)
        match_index = np.argmin(face_dis)

        if matches[match_index]:
            name = person_names[match_index].upper()
            current_names.add(name)
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    for name in current_names:
        attendance(name)

    frame_window.image(frame, channels='BGR')
    if cv2.waitKey(1) == 13 or not st.session_state.run:
        break

# Ensure camera is released if the session is stopped
if 'cap' in st.session_state and st.session_state.cap.isOpened():
    st.session_state.cap.release()
    cv2.destroyAllWindows()
