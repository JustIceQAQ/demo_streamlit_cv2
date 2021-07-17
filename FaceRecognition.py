import os
import streamlit as st

from haarcascades_unit import HaarCascades

st.title("互動式圖像人臉辨識")

uploaded_file = st.file_uploader("請上傳圖片", type=['png', 'jpg'])
col1, col2 = st.beta_columns(2)
col3, col4 = st.beta_columns(2)

if uploaded_file is not None:
    image_load_state = st.text('正在辨識圖片...')
    bytes_data = uploaded_file.getvalue()
    cv2_image = HaarCascades.read_image(bytes_data).to_cv2_haarcascades()
    dlib_image = HaarCascades.read_image(bytes_data).to_dlib_haarcascades()

    cv2_smile_image = HaarCascades.read_image(bytes_data).to_cv2_smile()
    cv2_eye_image = HaarCascades.read_image(bytes_data).to_cv2_eye()

    with col1:
        st.header(
            """Used CV2
        haarcascade frontalface default
        """)
        st.image(cv2_image)
    with col2:
        st.header(
            """Used dlib
        predictor 68 face landmarks
        """)
        st.image(dlib_image)
    with col3:
        st.header(
            """Used CV2
        haarcascade smile
        """)
        st.image(cv2_smile_image)
    with col4:
        st.header(
            """Used CV2
        haarcascade eye
        """)
        st.image(cv2_eye_image)

    image_load_state.text("")
    # st.balloons()

uploaded_video_file = st.file_uploader("請上傳影片", type=['mp4'])
data_load_state = st.text('')
col5, col6 = st.beta_columns(2)
if uploaded_video_file is not None:
    data_load_state = st.text('正在辨識影片...')
    if os.path.exists("cv2_output.mp4"):
        os.remove("cv2_output.mp4")
        col5_video = st.empty()
    if os.path.exists("dlib_output.mp4"):
        os.remove("dlib_output.mp4")
        col5_video = st.empty()

    video_bytes = uploaded_video_file.read()
    haarcascades_check = HaarCascades.read_video(video_bytes).used_cv2_haarcascades()
    dlib_check = HaarCascades.read_video(video_bytes).used_dlib_haarcascades()

    if haarcascades_check:
        with col5:
            video_file = open('cv2_output.mp4', 'rb')
            cv2_output_data = video_file.read()
            col5_video = st.video(cv2_output_data)
    if dlib_check:
        with col6:
            dlib_video_file = open('dlib_output.mp4', 'rb')
            col6_video = st.video(dlib_video_file)
    data_load_state.text('')
    st.balloons()
else:
    if os.path.exists("cv2_output.mp4"):
        with col5:
            video_file = open('cv2_output.mp4', 'rb')
            cv2_output_data = video_file.read()
            st.video(cv2_output_data)
    if os.path.exists("dlib_output.mp4"):
        with col6:
            video_file = open('dlib_output.mp4', 'rb')
            st.video(video_file)
    data_load_state.text('')
