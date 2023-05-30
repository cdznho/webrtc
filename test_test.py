'''
This script gets a model from the model folder and makes a prediction. It was trained on self-made images and is now
'''


import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
#import time
from cvzone.ClassificationModule import Classifier

#the 0 is the ID of the webcam on the computer
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
model_path = "model/sign_language/keras_model.h5"
labels_path = "model/sign_language/labels.txt"

classifier = Classifier(model_path, labels_path)

offset = 40
imgSize = 300

FOLDER = "Data/C"
counter = 0

labels = ["A", "B", "C"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        #if we don't add this, the pics can be rectangles or squares of diff sizes. uint8 - unsigned integer of 8 bits gives up a range from 0 to 255.
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y+h + offset, x - offset:x+w +offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        # Fixed height, recalculate width
        if aspectRatio > 1:
            # if we stretch out to imgsize, then what is the new width? We need to calculate the constant k for this
            k = imgSize/h
            wCal = math.ceil(k*w)
            #width before the height
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            # centering the pic. How much should we push the image forward to be able to center it? = wGap
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:,wGap: wCal+wGap] = imgResize

        # Fixed width, recalculate height
        else:
            # if we stretch out to imgsize, then what is the new width? We need to calculate the constant k for this
            k = imgSize / w
            hCal = math.ceil(k*h)
            #width before the height
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            # centering the pic. How much should we push the image forward to be able to center it? = wGap
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize

        prediction, idx = classifier.getPrediction(imgWhite, draw=False)
        print(prediction, idx)

        cv2.rectangle(imgOutput, (x - offset, y -offset-50), (x-offset+90,y-offset),(255,0,255), cv2.FILLED)
        cv2.putText(imgOutput, labels[idx], (x,y-offset), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset,y- offset), (x+w+offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)
