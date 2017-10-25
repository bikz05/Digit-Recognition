#!/usr/bin/python

# Import the modules
import cv2
from keras.models import load_model
import numpy as np
import argparse as ap


def show_image(im):
    cv2.namedWindow("Resulting Image with Rectangular ROIs", cv2.WINDOW_NORMAL)
    cv2.imshow("Resulting Image with Rectangular ROIs", im)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()


# Get the path of the training set
parser = ap.ArgumentParser()
# parser.add_argument("-c", "--classiferPath", help="Path to Classifier File", required="True")
parser.add_argument("-i", "--image", help="Path to Image", required="True")
args = vars(parser.parse_args())

# Load the classifier
# clf, pp = joblib.load(args["classiferPath"])
model = load_model('digit_model.h5')

# Read the input image
im = cv2.imread(args["image"])
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

im_for_ctr = cv2.GaussianBlur(im_gray, (5, 5), 0)
im_th = cv2.Canny(im_for_ctr, 100, 200)
# show_image(im_th)

# Find contours in the image
ret, binary = cv2.threshold(im_gray, 210, 255, cv2.THRESH_BINARY)

_, ctrs, _ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]


for rect in rects:
    if 15 < rect[2] < 50 and 25 < rect[3] < 50:
        pass
    else:
        continue

    if 440 < rect[0] < 620 and 125 < rect[1] < 165:
        pass
    else:
        continue

    # Draw the rectangles
    # rect[0], rect[1] : top left coor , rect[2], rect[3] : width and height
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.2)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = binary[pt1:pt1+leng, pt2:pt2+leng]
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    # show_image(roi)

    roi = np.array(roi, dtype='float32')
    roi = roi.reshape((1, 28, 28, 1))

    # predict
    nbr = model.predict(roi)
    result = np.argmax(nbr)

    cv2.putText(im, str(result), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

cv2.namedWindow("Resulting Image with Rectangular ROIs", cv2.WINDOW_NORMAL)
cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey(10000)
