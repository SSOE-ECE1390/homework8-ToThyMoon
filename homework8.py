import easyocr.export
import numpy as np
import cv2
import os
import glob
import easyocr
import pytesseract
import mediapipe as mp
from typing import Tuple, Union
import math

def calibrate():

    filename = os.path.relpath('./Calibration Images/Calibration')
    num_images = 1

    img_filenames = glob.glob(os.path.relpath(filename + "*.jpg"))
    img_filenames=sorted(img_filenames)

    img_list = [cv2.imread(file) for file in img_filenames]

    img_list=[cv2.cvtColor(img_list[idx],cv2.COLOR_BGR2GRAY) for idx in range(0,len(img_list))]

    img_corners=[]
    obj_corners=[]

    CHECKERBOARD = (7,10)  # The size is number of corners, 
    # which is one less than the number of squares

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    for idx in range(0,len(img_list)):
        retval, corners = cv2.findChessboardCorners(img_list[idx],patternSize =CHECKERBOARD)
        if(retval):
            corners=cv2.cornerSubPix(img_list[idx], corners, (11,11),(-1,-1), criteria)
            img_corners.append(corners)
            obj_corners.append(objp)
            img_list[idx] = cv2.drawChessboardCorners(img_list[idx], CHECKERBOARD, corners, retval)
        
    # print(img_list[0].shape)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_corners, img_corners, img_list[0].shape[::-1], None, None)
    
    print("Camera matrix : \n")
    print(mtx)
    print("distortion parameters : \n")
    print(dist)

    return mtx, dist

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()
  height, width, _ = image.shape

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    # Draw keypoints
    for keypoint in detection.keypoints:
      keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                     width, height)
      color, thickness, radius = (0, 255, 0), 2, 2
      cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Draw label and score
    for idx in range(0,len(detection.categories)):
      category = detection.categories[idx]
      category_name = category.category_name
      category_name = '' if category_name is None else category_name
      probability = round(category.score, 2)
      result_text = category_name + ' (' + str(probability) + ')'
      text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
      cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return annotated_image
img = cv2.imread("./image.jpg")

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

detector = cv2.aruco.ArucoDetector(dictionary)
# cv2.Rodrigues()

corners, ids, rejected = detector.detectMarkers(img)

for idx in range(0,ids.shape[0]):
    thiscorner=corners[idx][0]
    # print(thiscorner)
    pos=np.uint16(thiscorner.mean(axis=0))
    # print(pos)
    cv2.putText(img,f"{ids[idx]}",pos,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    for j in range(thiscorner.shape[0]):
        cor = thiscorner[j]
        # print(cor)
        cv2.drawMarker(img, np.uint16(cor), (0,50*j,0), thickness=10)

bottom_left = np.uint16(corners[0][0][3])
bottom_right = np.uint16(corners[1][0][2])
top_right = np.uint16(corners[2][0][1])
# print(f"bottom left: {bottom_left}")
# print(f"bottom_right: {bottom_right}")
# print(f"top_right: {top_right}")

# cv2.drawMarker(img, np.uint16(top_right), (0,0,255), thickness=10)
top_left = np.array([bottom_left[0] - (bottom_left[0]-bottom_right[0]), top_right[1] - (bottom_left[1] - bottom_right[1])])
middle = np.array([(bottom_right[0] - bottom_left[0])/2, (bottom_right[1] - top_right[1])/2], dtype=np.uint16)

cv2.drawMarker(img, np.uint16(top_left), (0,0,255), thickness=10)
cv2.drawMarker(img, np.uint16(middle), (0,0,255), thickness=10)

# Couldn't figure out how to align/orient with the ARUCO markers
'''
mtx, dist = calibrate()
rvecs, tvecs, _ = my_estimatePoseSingleMarkers(corners, 4, mtx, dist)

# Extract rotation angles (assuming you want to use the first marker)
rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
print(rotation_matrix)
euler_angles = cv2.decomposeProjectionMatrix(rotation_matrix)[1]

# Create rotation matrix based on the angles
rotation_matrix = cv2.getRotationMatrix2D((img.shape[1], img.shape[0]), euler_angles[1], 1)

# Apply rotation to the image
rotated_img = cv2.warpPerspective(img, img, rotation_matrix)
'''

# Name
name_img = img[300:500, 400:1500,:]
name_img_gray = cv2.cvtColor(name_img, cv2.COLOR_BGR2GRAY)
reader = easyocr.Reader(['en'])
result = reader.readtext(name_img_gray, paragraph='False')
print(result)

# Class Check
img_1390 = img[750:850, 350:550,:]
img_2390 = img[980:1080, 350:550,:]

# cv2.imshow("1390", img_1390)
# cv2.imshow("2390", img_2390)

img_1390_gray = cv2.cvtColor(img_1390, cv2.COLOR_BGR2GRAY)
img_2390_gray = cv2.cvtColor(img_2390, cv2.COLOR_BGR2GRAY)
mean_1390 = cv2.mean(img_1390_gray)
mean_2390 = cv2.mean(img_2390_gray)

class_num = "1390" if mean_1390 < mean_2390 else "2390"

print(f"Class: ECE{class_num}")


# Numbers
num_img = img[1340:1500, 460:1320, :]
num_img_gray = cv2.cvtColor(num_img, cv2.COLOR_BGR2GRAY)


# Text
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
text_img = img[1950:2450, 300:1650,:]
text_img_gray = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
data = pytesseract.image_to_data(text_img_gray, output_type=pytesseract.Output.DICT)
text_tot = ""
# print(data)
cv2.imshow("",text_img_gray)
if(not data.__class__==str):
    for i in range(len(data["text"])):
        if data["text"][i] != "":
            x=data['left'][i]
            y=data['top'][i]
            w=data['width'][i]
            h=data['height'][i]
            text=data['text'][i]
            text_tot += text

print(f"Text: {text_tot}")

# Face
face_img = np.uint16(img[1500:2620,1800:2700,:])
face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the image mode:
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.IMAGE)

mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA64, data=face_img)
detector=FaceDetector.create_from_options(options)

face_detector_result = detector.detect(mp_image)
face_img_anno=visualize(mp_image.numpy_view(),face_detector_result)
# cv2.imshow("face", face_img_anno)
cv2.imwrite("face_marked.jpg", face_img_anno)

# cv2.imwrite("markers.jpg", img)
# cv2.imwrite("rotated.jpg", rotated_img)
# cv2.imshow("x",img)
# cv2.imshow("name", name_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

