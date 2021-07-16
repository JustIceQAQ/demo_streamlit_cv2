import streamlit as st

from haarcascades_unit import HaarCascades

st.title("互動式圖像人臉辨識")

uploaded_file = st.file_uploader("請上傳圖片")
col1, col2 = st.beta_columns(2)

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    cv2_image = HaarCascades.read_image(bytes_data).to_cv2_haarcascades()
    dlib_image = HaarCascades.read_image(bytes_data).to_dlib_haarcascades()

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
