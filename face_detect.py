import cv2
import sys
import time
import random

from skimage.feature import local_binary_pattern

# Get Haas cascade file data
cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

# Initialize webcam (by default use camera with index 0)
video_capture = cv2.VideoCapture(0)

# Initialize defaults
start_showing_face_time = 0
start_showing_text_time = 0
text_index = 0
color_index = 0
detected_face = False
WAIT_UNTIL_TEXT = 0 # sec
SHOW_TEXT_TIME = 5 # sec

# Read text and color files
textes = [text.strip() for text in open('textes.txt', 'r').readlines() if text.strip()]
colors = [color.strip().split(',') for color in open('colors.txt', 'r').readlines() if color.strip()]
colors = [[int(c) for c in color] for color in colors]


while True:
    # Capture frame-by-frame from web cam
    ret, frame = video_capture.read()

    # Transform frame to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply masc from haas cascade to frame
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) and not detected_face:
        print 'Detected face'
        detected_face = True
        start_showing_face_time = time.time()
    elif not len(faces) and detected_face:
        print 'Loose face'
        detected_face = False

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (189,183,107), 1)

        if detected_face and time.time() - start_showing_face_time > WAIT_UNTIL_TEXT:
            if time.time() - start_showing_text_time > SHOW_TEXT_TIME:
                text_index = random.randrange(len(textes))
                color_index = random.randrange(len(colors))
                start_showing_text_time = time.time()
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, textes[text_index], (x - 5, y - 10), font, 1, colors[color_index], 3)

        lbp = local_binary_pattern(cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY), 2*3, 2, method='uniform')
        cv2.imshow('win name', lbp)
        cv2.waitKey(1)


    # Display the resulting frame
    cv2.imshow('Video', frame)
    # time.sleep(0.2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()