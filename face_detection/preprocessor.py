import cv2
import numpy as np
from PIL import Image, ImageEnhance
import imutils


def preprocess_id_image_from_memory(file):
    file.seek(0)
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original = image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise and reduce lighting impact
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    background = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    diff = 255 - cv2.absdiff(denoised, background)
    enhanced = cv2.equalizeHist(diff)

    # Find contours to align image
    edged = cv2.Canny(enhanced, 30, 200)
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    screen_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen_cnt = approx
            break

    if screen_cnt is not None:
        warped = four_point_transform(original, screen_cnt.reshape(4, 2))
    else:
        warped = original

    # Convert to PIL and enhance
    pil_img = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(1.2)
    pil_img = ImageEnhance.Brightness(pil_img).enhance(1.05)

    return pil_img  # You can save or convert this as needed


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0], [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))
