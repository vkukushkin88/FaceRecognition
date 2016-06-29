import os
import cv2
import sys
import time
import pickle
import random
import argparse

import numpy as np

from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq


# Initialize defaults
RECOGNISHN_SCORE = 0.01

RADIUS = 4
NUM_POINTS = 4 * RADIUS

DUMP_FILE = './dump_faces.lbph'


class FaceRecognizer(object):

    def __init__(self, args):
        self.args = args

        # Get Haas/RBD cascade file data
        # In current implementation this is no matter what to use
        self.faceCascade = cv2.CascadeClassifier(self.args.cascade_file)

        self.X_test = []
        self.X_name = []
        self.__load_dump()
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self._colors = [color.strip().split(',') for color in open('colors.txt', 'r').readlines() if color.strip()]
        self._colors = [[int(c) for c in color] for color in self._colors]

    def process(self):
        while True:
            # Capture frame-by-frame from web cam
            ret, frame = self.video_capture.read()

            are_detections, faces = self.detect_face(frame)

            if are_detections:
                # Face(s) are detected, next step is to recognize it
                # and show why is it or ask about this people
                for (x, y, w, h) in faces:
                    coordinates = (x, y, w, h)
                    face = self.__crop_face(frame, x, y, w, h)
                    self.recognize_face(frame, face, coordinates)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (189,183,107), 1)

            # Display the resulting frame
            cv2.imshow('Video', frame)
            time.sleep(0.2)

            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break

    def detect_face(self, img):
        # Detect face using Haar/RBD cascade. Is not complex task.
        # We have already Haar/RBD cascade and only we need to do is apply
        # cascade to frame from camera

        # Transform img to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply masc from Haas/RBD cascade to img
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return len(faces), faces

    def recognize_face(self, frame, face, coordinates):
        # Recognizing is second part of our tool. It is more complex that
        # simple detection. This recognizing is based on
        # Local Binary Patterns Histograms (LBPH) method
        # Calculate the histogram
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(face_gray, NUM_POINTS, RADIUS, method='uniform')
        x = itemfreq(lbp.ravel())
        # Normalize the histogram
        hist = x[:, 1] / sum(x[:, 1])

        for index, x in enumerate(self.X_test):
            score = cv2.compareHist(
                np.array(x, dtype=np.float32),
                np.array(hist, dtype=np.float32),
                cv2.HISTCMP_CHISQR
            )
            score = round(score, 3)
            print 'Score: ', score

            if score < RECOGNISHN_SCORE:
                # Face detected, show who is
                (x, y, w, h) = coordinates
                cv2.putText(
                    frame,
                    self.X_name[index],
                    (x - 5, y - 10),
                    self.font,
                    1,
                    self.__get_random_color(),
                    3
                )
                break
        else:
            win_name = 'Detected new Face, do you know why it is?'
            cv2.imshow(win_name, face)
            cv2.waitKey(500)
            self.ask_whois(hist)
            cv2.destroyWindow(win_name)

    def ask_whois(self, hist):
        name = raw_input('Detected new face, who is this? ')
        self.X_test.append(hist)
        self.X_name.append(name)

    def __crop_face(self, img, x, y, w, h):
        return img[y:y+h, x:x+w]

    def __get_random_color(self):
        return random.randrange(len(self._colors))

    def __save_dump(self):
        dump_file = open(DUMP_FILE, 'wb')
        pickle.dump((self.X_name, self.X_test), dump_file)
        dump_file.close()

    def __load_dump(self):
        if os.path.exists(DUMP_FILE):
            dump_file = open(DUMP_FILE, 'rb')
            try:
                self.X_name, self.X_test = pickle.load(dump_file)
            except Exception:
                # ignore incorrect reading information from file
                pass
            finally:
                dump_file.close()

    def run(self):
        # Initialize webcam (by default use camera with index 0)
        self.video_capture = cv2.VideoCapture(0)

        try:
            self.process()
        finally:
            # When everything is done, release the capture
            self.video_capture.release()
            cv2.destroyAllWindows()
            self.__save_dump()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cascade-file',
        help='Path to Cascade file with Haar or LBP pattern of face',
        required='True')
    return parser.parse_args()


if __name__ == '__main__':
    fr = FaceRecognizer(parse_args())
    fr.run()
