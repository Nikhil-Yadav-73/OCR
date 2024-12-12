import cv2
import pytesseract
import numpy as np
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            cropped = image[y:y + h, x:x + w]
            return cropped
    
    return None

def extract_text_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    text = pytesseract.image_to_string(thresh, config='--psm 8')
    return text.strip()

if __name__ == "__main__":
    image_path = r"C:\Users\Nikhil\Desktop\OCR\venv\Nik\np2.png"
    # i guess because of pytesserect engine
    cropped_plate = preprocess_image(image_path)
    
    if cropped_plate is not None:
        text = extract_text_from_image(cropped_plate)
        print(f"Detected Number Plate: {text}")
    else:
        print("Number plate nahi samajh paye.")
