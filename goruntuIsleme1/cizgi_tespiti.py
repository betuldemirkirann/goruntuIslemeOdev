import cv2
import numpy as np

# Görseli oku
image = cv2.imread('eyes.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)  # Gürültüyü azaltmak için median blur uygula

# Hough dönüşümü için dairelerin tespiti
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                           param1=50, param2=30, minRadius=10, maxRadius=50)

# Tespit edilen daireler varsa işaretleyelim
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 0, 255), 2)

# Sonucu göster
cv2.imshow('Eyes Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
