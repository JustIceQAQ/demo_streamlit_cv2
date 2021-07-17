import io
import os
import tempfile

import dlib
import numpy as np
import cv2


class HaarCascades:

    def __init__(self, binary_file=None):
        self.CV2_HAARCASCADE_FRONTALFACE_DEFAULT = "./model/haarcascade_frontalface_default.xml"
        self.CV2_HAARCASCADE_SMILE = "./model/haarcascade_smile.xml"
        self.CV2_HAARCASCADE_EYE = "./model/haarcascade_eye.xml"
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

    def to_cv2_smile(self):
        classifier = cv2.CascadeClassifier(self.CV2_HAARCASCADE_FRONTALFACE_DEFAULT)
        smile_Cascade = cv2.CascadeClassifier(self.CV2_HAARCASCADE_SMILE)
        face_rects = classifier.detectMultiScale(
            self.binary_file, scaleFactor=1.2, minNeighbors=4, minSize=(24, 24))
        if len(face_rects):
            for faceRect in face_rects:
                x, y, w, h = faceRect
                # cv2.rectangle(self.binary_file, (x, y), (x + h, y + w), self.USED_COLOR, self.LINE_WIDE)
                roi_gray = self.binary_file[y:y + h, x:x + w]
                roi_color = self.binary_file[y:y + h, x:x + w]
                smile = smile_Cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.16,
                    minNeighbors=35,
                    minSize=(25, 25),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                for (x2, y2, w2, h2) in smile:
                    cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), self.LINE_WIDE)

        return self.binary_file

    def to_cv2_eye(self):
        classifier = cv2.CascadeClassifier(self.CV2_HAARCASCADE_EYE)
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

    @classmethod
    def read_video(cls, binary_file):
        t_file = tempfile.NamedTemporaryFile(delete=False)
        t_file.write(binary_file)
        vf = cv2.VideoCapture(t_file.name)
        class_method = cls(vf)
        t_file.close()
        return class_method

    def used_cv2_haarcascades(self):
        cascade = cv2.CascadeClassifier(self.CV2_HAARCASCADE_FRONTALFACE_DEFAULT)
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter('cv2_output.mp4', fourcc, 15.0, (640, 360))
        flag = 0
        while True:
            # Capture frame-by-frame
            ret, img = self.binary_file.read()
            if ret == True:
                # converting to gray image for faster video processing
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4,
                                                 minSize=(50, 50))
                # if at least 1 face detected
                if len(rects) >= 0:
                    # Draw a rectangle around the faces
                    for (x, y, w, h) in rects:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if flag % 5 == 0:
                    out.write(img)
                    flag += 1
                else:
                    flag += 1
            else:
                break
        out.release()
        return True

    def used_dlib_haarcascades(self):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            self.DLIB_SHAPE_PREDICTOR_68_FACE_LANDMARKS
        )
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter('dlib_output.mp4', fourcc, 15.0, (640, 360))
        flag = 0
        while True:
            # Capture frame-by-frame
            ret, img = self.binary_file.read()
            if ret:
                # converting to gray image for faster video processing
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                dets = detector(gray, 1)
                if flag % 25 == 0:
                    for face in dets:
                        shape = predictor(img, face)
                        for pt in shape.parts():
                            pt_pos = (pt.x, pt.y)
                            cv2.circle(img, pt_pos, 3, self.USED_COLOR, 1)
                    out.write(img)
                    flag += 1
                else:
                    flag += 1
            else:
                break
        out.release()
        return True
