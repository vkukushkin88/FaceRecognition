# FaceRecognition
[EDU] Self education in face recognition technologies

There are available two scripts:

 - face_detect.py     - simple script which uses OpenCV lib to detect face coming from Web Camera
 - face_recognizer.py - simple script whicn can detect and "recognize" detected face coming from Web Camera


# Dependencies
OpenCV 3.0.0.0 or never
NumPy 1.11.0 or never
skimage 0.12.3
scipy 0.17.1


# Installing
To run thin you should install all dependencies

## How to install dependencies:
 - How to install OpenCV you can read [here](http://opencv.org/opencv-3-0.html), especial for Ubuntu (Linux) you can get instructions [here](http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/)
 - How to install [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
 - How to install [skimage](http://scikit-image.org/docs/dev/install.html)
 - How to install [scipy](https://www.scipy.org/install.html)

# Description:
Process of face recognizing is not an easy proceed. Currently exists a lot of different methods to do this.
In current project was selected OpenCV as base lib for processing video stream and Local Binary Pattern Histogram method for recognize object.

Process of recognizing is divided by following steps:

 - capture frame from web camera
 - detect face
 - recognizing:
   * compare with knowing histograms
   * if found similar face than show it
   * if face is unknown than ask who is on face


# Running applications
To run application please make sure you have installed all packages and plugged in Web Camera

Running:
> $ cd `<your_working_directory>`

> $ python face_recognizer.py -c lbpcascade_frontalface.xml


## Note:
Face detection and recognize are different things.
Detection is only find and mark somehow object. Recognizing is more complicated thing.
`face_detect.py` script is done only for showing same detection process without any additional.

## Note:
This example shows how to detect faces but something similar can be done for any object. Only you need to find appropriate cascade file.

 
