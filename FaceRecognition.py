import io

import numpy as np
import streamlit as st
import cv2
import dlib

CV2_HAARCASCADE_FRONTALFACE_DEFAULT = "./model/haarcascade_frontalface_default.xml"
DLIB_SHAPE_PREDICTOR_68_FACE_LANDMARKS = "./model/shape_predictor_68_face_landmarks.dat"
COLOR = (0, 255, 0)
WIDE = 4


# https://github.com/davisking/dlib-models
# https://github.com/opencv/opencv/tree/master/data

def cv2_haarcascades(image):
    classifier = cv2.CascadeClassifier(CV2_HAARCASCADE_FRONTALFACE_DEFAULT)
    faceRects = classifier.detectMultiScale(
        image, scaleFactor=1.2, minNeighbors=4, minSize=(24, 24))
    if len(faceRects):
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv2.rectangle(image, (x, y), (x + h, y + w), COLOR, WIDE)
    return image


def dlib_haarcascades(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        DLIB_SHAPE_PREDICTOR_68_FACE_LANDMARKS
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    for face in dets:
        shape = predictor(image, face)
        for pt in shape.parts():
            pt_pos = (pt.x, pt.y)
            cv2.circle(image, pt_pos, 4, COLOR, WIDE)
    return image


def to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


st.title("互動式圖像人臉辨識")

uploaded_file = st.file_uploader("請上傳圖片")
col1, col2 = st.beta_columns(2)

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    image_stream = io.BytesIO()
    image_stream.write(bytes_data)
    image_stream.seek(0)

    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)

    image1 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    image1RGB = to_rgb(image1)
    image2RGB = to_rgb(image2)

    cv2_image = cv2_haarcascades(image1RGB)
    dlib_image = dlib_haarcascades(image2RGB)

    with col1:
        st.header("""Used CV2
        haarcascade frontalface default
        """)
        st.image(cv2_image)
    with col2:
        st.header("""Used dlib
        predictor 68 face landmarks
        """)
        st.image(dlib_image)
