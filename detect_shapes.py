from shapedetector import ShapeDetector
import imutils
import cv2
import numpy as np

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread("shapes_and_colors.jpg")
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly, and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# find contours in the thresholded image and initialize the shape detector
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()

# loop over the contours
for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	shape = sd.detect(c)

	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
             0.5, (255, 255, 255), 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)

# then try to test webcam
# vid = cv2.VideoCapture(0)
# while (True):
#     ret, frame = vid.read()
#     if ret == True:
#         # load the image and resize it to a smaller factor so that
#         # the shapes can be approximated better
#         image = frame
#         resized = imutils.resize(image, width=300)
#         ratio = image.shape[0] / float(resized.shape[0])
#         # convert the resized image to grayscale, blur it slightly,
#         # and threshold it
#         gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#         cv2.imshow("gray", gray)
#         blurred = cv2.GaussianBlur(gray, (3, 3), 0)
#         thresh = cv2.threshold(blurred, 60, 220, cv2.THRESH_BINARY)[1]
#         cv2.imshow("thresh", thresh)
#         # find contours in the thresholded image and initialize the
#         # shape detector
#         cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
#                                 cv2.CHAIN_APPROX_SIMPLE)
#         cnts = imutils.grab_contours(cnts)
#         sd = ShapeDetector()
#         # loop over the contours
#         for c in cnts:
#             # compute the center of the contour, then detect the name of the
#             # shape using only the contour
#             M = cv2.moments(c)
#             if M["m00"] != 0:
#                 cX = int((M["m10"] / M["m00"]) * ratio)
#                 cY = int((M["m01"] / M["m00"]) * ratio)
#             else:
#                 cX = 0
#                 cY = 0
#             shape = sd.detect(c)
#             # multiply the contour (x, y)-coordinates by the resize ratio,
#             # then draw the contours and the name of the shape on the image
#             c = c.astype("float")
#             c *= ratio
#             c = c.astype("int")
#             cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
#             cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5, (255, 255, 255), 2)
#             # show the output image
#             # cv2.imshow("Image", image)
#             # cv2.waitKey(0)
#         cv2.imshow("Shape_Detect", frame)
    
#     if cv2.waitKey(20) == 27:
#         break

# vid.release()
# cv2.destroyAllWindows()
