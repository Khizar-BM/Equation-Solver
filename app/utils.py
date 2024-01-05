import numpy as np
from model import load_model
import cv2
from imutils.contours import sort_contours
import numexpr as ne

model = load_model()
labels = ['%',
          '*',
          '+',
          '-',
          '0',
          '1',
          '2',
          '3',
          '4',
          '5',
          '6',
          '7',
          '8',
          '9',
          '[',
          ']']

labels_dict = dict(enumerate(labels))


def preprocess_img(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarize the image
    _, binary_image = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return binary_image


def detect_contours(image, binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sort_contours(contours, method="left-to-right")[0]

    chars = []
    result = []

    for contour in contours:
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 1200:
            # Extract and resize character
            char_image = binary_image[y:y + h + 10, x:x + w + 10]
            char_image = cv2.resize(char_image, (28, 28))
            # eroded_image = cv2.erode(char_image, (3, 3), iterations=1)

            chars.append(~char_image)
            result.append(make_prediction(~char_image))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return chars, result, image


def make_prediction(img):
    img = np.expand_dims(img, axis=0)  # Reshape for the model
    # Predict
    prediction = model.predict(img)
    pred_class = np.argmax(prediction, axis=1)
    return labels_dict[pred_class[0]]


def evaluate(expr):
    try:
        return float(ne.evaluate(''.join(expr).replace("%", "/")).squeeze())
    except:
        return "The expression could not be evaluated"
