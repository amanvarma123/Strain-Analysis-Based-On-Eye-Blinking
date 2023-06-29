# Import all the necessary packages required
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import datetime
from time import sleep

def distance_ratio(eye):
	# Compute the euclidean distances b/w the two sets of vertical (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
 
	# Compute the euclidean distance b/w the horizontal (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
 
	# Now Compute Eye aspect ratio
	eye_ratio = (A + B) / (2.0 * C)
 
	return eye_ratio
 
# Here Initialize the dlib's face detector and Create the facial landmark predictor

face_detect = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
eye_ratio = 0

def calculate_eye_ratio(frame, gray):

	# Grabbing the indexes of the facial landmarks for the left and right eye

	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

	points = face_detect(gray, 0)

	# Now loop over the face detections
	for r in points:
		# Determine the facial landmarks and convert the (x, y)-coordinates to a NumPy array
		shape = predictor(gray, r)
		shape = face_utils.shape_to_np(shape)

		l_eye = shape[lStart:lEnd]
		r_eye = shape[rStart:rEnd]
		l_eye_ratio = distance_ratio(l_eye)
		r_eye_ratio = distance_ratio(r_eye)
 
		# Calculate the average of eye aspect ratio for both eyes
		eye_ratio = (l_eye_ratio + r_eye_ratio) / 2.0

		# Finally Compute the convex hull for the left and right eye and visualize each of the eyes
		l_eyeHull = cv2.convexHull(l_eye)
		rightEyeHull = cv2.convexHull(r_eye)
		cv2.drawContours(frame, [l_eyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		return eye_ratio