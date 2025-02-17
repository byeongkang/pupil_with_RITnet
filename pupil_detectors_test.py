import cv2
from pupil_detectors import Detector2D

detector = Detector2D()

# read image as numpy array from somewhere, e.g. here from a file
img = cv2.imread("pupil.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

result = detector.detect(gray)
ellipse = result["ellipse"]

# draw the ellipse outline onto the input image
# note that cv2.ellipse() cannot deal with float values
# also it expects the axes to be semi-axes (half the size)
cv2.ellipse(
   img,
   tuple(int(v) for v in ellipse["center"]),
   tuple(int(v / 2) for v in ellipse["axes"]),
   ellipse["angle"],
   0, 360, # start/end angle for drawing
   (0, 0, 255) # color (BGR): red
)

print(result)
cv2.imshow("Image", img)
cv2.waitKey(0)

