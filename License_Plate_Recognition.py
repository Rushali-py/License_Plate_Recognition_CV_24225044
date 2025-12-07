import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import pytesseract

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 1. Read image
IMAGE_PATH = "image9.jpg"   # change to your file name
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
plt.show()

# 2. Sharpen + contrast (helps a bit for blur)
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
sharp = cv2.filter2D(gray, -1, kernel)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(sharp)

# 3. Bilateral filter + Canny edges
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(bfilter, 30, 200)
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
plt.show()

# 4. Find plate location from contours, with fallback
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

h, w = gray.shape

if location is None:
    print("No plate contour found, using center crop instead.")
    # fallback: approximate plate region in lower-center part of image
    y1, y2 = int(h * 0.55), int(h * 0.9)
    x1, x2 = int(w * 0.2), int(w * 0.8)
    mask = np.zeros(gray.shape, np.uint8)
    mask[y1:y2, x1:x2] = 255
    new_image = cv2.bitwise_and(img, img, mask=mask)
    location = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]])
else:
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

# 5. Crop plate region
(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
plt.imshow(cropped_image, cmap="gray")
plt.show()

# 6. OCR with Tesseract
crop_bin = cv2.threshold(
    cropped_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)[1]
crop_bin = cv2.medianBlur(crop_bin, 3)

custom_config = r"--oem 3 --psm 7"
text = pytesseract.image_to_string(crop_bin, config=custom_config).strip()
print("Detected text:", text if text else "NOT_FOUND")

# 7. Draw result
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(
    img, text=text,
    org=(location[0][0][0], location[1][0][1] + 60),
    fontFace=font, fontScale=1,
    color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA,
)
res = cv2.rectangle(
    img, tuple(location[0][0]), tuple(location[2][0]),
    (0, 255, 0), 3
)

plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.show()
