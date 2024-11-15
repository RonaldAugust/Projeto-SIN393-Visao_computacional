import cv2
from skimage import transform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def segment_image(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray = (gray * 255).astype(np.uint8)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    segmented = cv2.bitwise_and(gray, gray, mask=mask)
    return segmented

def calculate_hu_moments(image):
    moments = cv2.moments(image)
    if moments['m00'] == 0:
        return np.zeros(7)
    hu_moments = cv2.HuMoments(moments).flatten()
    return -np.sign(hu_moments) * np.log10(np.abs(hu_moments))

def calculate_contour_signature(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([0])
    cnt = max(contours, key=cv2.contourArea)
    moments = cv2.moments(cnt)
    if moments['m00'] == 0:
        return np.array([0])
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return np.array([np.linalg.norm(np.array([cx, cy]) - point[0]) for point in cnt])

def calculate_features(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {'area': 0, 'major_axis': np.nan, 'minor_axis': np.nan, 'solidity': 0, 'eccentricity': np.nan}
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    try:
        ellipse = cv2.fitEllipse(cnt)
        major_axis, minor_axis = max(ellipse[1]), min(ellipse[1])
    except:
        major_axis, minor_axis = np.nan, np.nan
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else np.nan
    return {'area': area, 'major_axis': major_axis, 'minor_axis': minor_axis, 'solidity': solidity, 'eccentricity': eccentricity}
