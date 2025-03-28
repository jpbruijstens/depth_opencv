# processing.py
import cv2
import numpy as np
import depthai as dai
import modules.sliders as sliders


def convert_to_grayscale(frame):
    """
    Converts a frame to grayscale.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def calc_histogram(frame):
    """
    Calculates the histogram of a frame.
    """
    return cv2.calcHist([frame], [0], None, [256], [0, 256])

def convert_to_hsv(frame):
    """
    Converts a BGR frame to HSV.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

def apply_gaussian_blur(frame, ksize=(5,5)):
    """
    Applies a Gaussian blur to a frame.
    """
    return cv2.GaussianBlur(frame, ksize, 0)

def detect_edges(frame, threshold1=100, threshold2=200):
    """
    Detects edges in a frame.
    """
    return cv2.Canny(frame, threshold1, threshold2)

def apply_threshold(frame, threshold=128, maxval=255, type=cv2.THRESH_BINARY):
    """
    Applies a threshold to a frame.
    """
    return cv2.threshold(frame, threshold, maxval, type)[1]

def apply_morphology(frame, iterations=1):
    """
    Applies morphological operations to a frame.
    """
    kernel = (5, 5)
    kernel = np.ones(kernel, np.uint8)
    return cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations=iterations)

def apply_contours(frame):
    """
    Finds contours in a frame.
    """
    return cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def draw_contours(frame, contours, color=(0, 255, 0), thickness=2):
    """
    Draws contours on a frame.
    """
    return cv2.drawContours(frame, contours, -1, color, thickness)

def draw_rectangle(frame, x, y, w, h, color=(0, 255, 0), thickness=2):
    """
    Draws a rectangle on a frame.
    """
    return cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)

def draw_text(frame, text, x, y, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 255, 0), thickness=2):
    """
    Draws text on a frame.
    """
    return cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

def apply_mask(frame):
    """
    Applies a mask to a frame.
    """
    lowerBound = np.array([0, 120, 120])
    upperBound = np.array([10, 255, 255])
    
    return cv2.inRange(frame, lowerBound, upperBound)

def apply_erosion(frame, iterations=1):
    """
    Applies erosion to a frame.
    """
    kernel = (5, 5)
    kernel = np.ones(kernel, np.uint8)
    return cv2.erode(frame, kernel, iterations=iterations)

def apply_dilation(frame,iterations=1):
    """
    Applies dilation to a frame.
    """
    kernel = (5, 5)
    kernel = np.ones(kernel, np.uint8)
    return cv2.dilate(frame, kernel, iterations=iterations)

def color_mask(frame, lower_hsv, upper_hsv):
    """
    Applies a color mask to a frame.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    # Je kunt meteen alleen de pixels binnen het masker tonen:
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    return result

def depth_of_object(depthFrameColor, spatialData, corners, center, spatialCalcConfigInQueue, config):
    """
    Calculates the depth of an region of interest in a frame.
    """
    for depthData in spatialData:
        if corners is not None:
            # new roi and update the config
            newRactangle = (int(center[0] - 10), int(center[1] - 10), 20, 20)

            config.roi = dai.Rect(
                newRactangle[0], newRactangle[1], newRactangle[2], newRactangle[3])
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.addROI(config)
            spatialCalcConfigInQueue.send(cfg)

            roi = depthData.config.roi
            roi = roi.denormalize(
                width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            printColor = (255, 255, 255)
            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), printColor, 1)
            cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 50), fontType, 0.5, printColor)
            cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 65), fontType, 0.5, printColor)
            cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 80), fontType, 0.5, printColor)

        return depthFrameColor

def bounding_box(frame, objects, depthFrame):
    """
    Draws a bounding box around objects in a frame.
    """
    center = None
    corners = None

    for x, y, w, h in objects:
        corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        center = (x + w // 2, (y + h // 2) - 20)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if center is not None:
            cv2.circle(depthFrame, center, 5, (255, 0, 0), -1)
            cv2.circle(frame, center, 5, (255, 0, 0), -1)
    return center, corners 

def filter_contours(frame):
    """
    Filters contours in a frame.
    """
    # Find contours in the mask to identify the red objects
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours and extract the bounding boxes of the remaining contours
    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2000:  # Filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            objects.append((x, y, w, h))

    return objects


def filter_frame(frame):
    # Convert the frame to HSV for color filtering
    hsv_frame = convert_to_hsv(frame)

    # Read slider positions (h_min, h_max, s_min, s_max, v_min, v_max)
    h_min, h_max, s_min, s_max, v_min, v_max = sliders.get_hsv_values("Filter")

    # Apply the color filter
    filtered_frame = color_mask(
        hsv_frame,
        (h_min, s_min, v_min),
        (h_max, s_max, v_max)
    )
    return filtered_frame
