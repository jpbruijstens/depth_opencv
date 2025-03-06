# sliders.py
import cv2

def create_hsv_sliders(window_name="HSV Sliders"):
    cv2.namedWindow(window_name)
    # Hue: 0-179, Saturation: 0-255, Value: 0-255 in OpenCV
    cv2.createTrackbar("H_min", window_name, 0, 179, lambda x: x)
    cv2.createTrackbar("H_max", window_name, 179, 179, lambda x: x)
    cv2.createTrackbar("S_min", window_name, 0, 255, lambda x: x)
    cv2.createTrackbar("S_max", window_name, 255, 255, lambda x: x)
    cv2.createTrackbar("V_min", window_name, 0, 255, lambda x: x)
    cv2.createTrackbar("V_max", window_name, 255, 255, lambda x: x)

def get_hsv_values(window_name="HSV Sliders"):
    h_min = cv2.getTrackbarPos("H_min", window_name)
    h_max = cv2.getTrackbarPos("H_max", window_name)
    s_min = cv2.getTrackbarPos("S_min", window_name)
    s_max = cv2.getTrackbarPos("S_max", window_name)
    v_min = cv2.getTrackbarPos("V_min", window_name)
    v_max = cv2.getTrackbarPos("V_max", window_name)
    return (h_min, h_max, s_min, s_max, v_min, v_max)
