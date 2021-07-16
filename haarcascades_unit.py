import io

import dlib
import numpy as np
import cv2


class HaarCascades:

    def __init__(self, binary_file=None):
        self.CV2_HAARCASCADE_FRONTALFACE_DEFAULT = "./model/haarcascade_frontalface_default.xml"
        self.DLIB_SHAPE_PREDICTOR_68_FACE_LANDMARKS = "./model/shape_predictor_68_face_landmarks.dat"
        self.USED_COLOR = (0, 255, 0)
        self.LINE_WIDE = 4
        self.binary_file = binary_file

    @classmethod
    def read_image(cls, binary_file):
        # BytesIO input
        image_stream = io.BytesIO()
        image_stream.write(binary_file)
        image_stream.seek(0)

        # as array
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        image_object = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # to RGB
        image_object_rgb = cv2.cvtColor(image_object, cv2.COLOR_BGR2RGB)

        class_method = cls(image_object_rgb)
        return class_method

    def to_cv2_haarcascades(self):
        classifier = cv2.CascadeClassifier(self.CV2_HAARCASCADE_FRONTALFACE_DEFAULT)
        face_rects = classifier.detectMultiScale(
            self.binary_file, scaleFactor=1.2, minNeighbors=4, minSize=(24, 24))
        if len(face_rects):
            for faceRect in face_rects:
                x, y, w, h = faceRect
                cv2.rectangle(self.binary_file, (x, y), (x + h, y + w), self.USED_COLOR, self.LINE_WIDE)
        return self.binary_file

    def to_dlib_haarcascades(self):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            self.DLIB_SHAPE_PREDICTOR_68_FACE_LANDMARKS
        )
        gray = cv2.cvtColor(self.binary_file, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 1)
        for face in dets:
            shape = predictor(self.binary_file, face)
            for pt in shape.parts():
                pt_pos = (pt.x, pt.y)
                cv2.circle(self.binary_file, pt_pos, 4, self.USED_COLOR, self.LINE_WIDE)
        return self.binary_file
