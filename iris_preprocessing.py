import cv2

iris = cv2.imread("left_iris_crop.jpg")

# Grayscale
gray = cv2.cvtColor(iris, cv2.COLOR_BGR2GRAY)

# Resize
resized = cv2.resize(gray, (64, 64))

# Histogram Equalization
equalized = cv2.equalizeHist(resized)

# Blur
processed = cv2.GaussianBlur(equalized, (3,3), 0)

cv2.imwrite("left_iris_processed.jpg", processed)
print("✅ Iris preprocessed and saved as left_iris_processed.jpg")
