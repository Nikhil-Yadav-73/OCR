import cv2
import pytesseract
import numpy as np
import os

# Configure Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the image.")
        exit()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Perform morphological operations to close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Look for rectangular contour
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            cropped = image[y:y + h, x:x + w]
            return cropped

    return None

def extract_text_from_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding for better OCR
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Configure Tesseract OCR
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    # Extract text using Tesseract
    text = pytesseract.image_to_string(thresh, config=custom_config)
    return text.strip()

def post_process_text(text):
    replacements = {
        'O': '0',
        'I': '1',
        '|': '1',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text

if __name__ == "__main__":
    image_path = r"C:\Users\Nikhil\Desktop\OCR\venv\Nik\np4.png"
    cropped_plate = preprocess_image(image_path)

    if cropped_plate is not None:
        text = extract_text_from_image(cropped_plate)
        processed_text = post_process_text(text)
        print(f"Detected Number Plate: {processed_text}")
    else:
        print("Number plate nahi samajh paye.")
