import cv2
import numpy as np
import pandas as pd

image = cv2.imread('say.jpg')

# Görüntüyü hiperspektral renk uzayından RGB'ye dönüştürün
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Koyu yeşil bölgeleri tespit edin
lower_green = np.array([0, 50, 0], dtype=np.uint8)
upper_green = np.array([50, 255, 50], dtype=np.uint8)
mask = cv2.inRange(rgb_image, lower_green, upper_green)

# Koyu yeşil bölgelerin kontürlerini bulun
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Excel tablosu için veri listesi oluşturun
data = []
for i, contour in enumerate(contours, start=1):
    # Konturun alanını hesaplayın
    area = cv2.contourArea(contour)
    if area > 0:
        # Konturun sınırlayıcı kutusunu hesaplayın
        x, y, w, h = cv2.boundingRect(contour)
        # Ortalama ve medyan renk değerlerini hesaplayın
        mean_color = np.mean(rgb_image[y:y+h, x:x+w], axis=(0, 1))
        median_color = np.median(rgb_image[y:y+h, x:x+w], axis=(0, 1))
        # Veri listesine ekle
        data.append([i, (x + w // 2, y + h // 2), w, h, np.sqrt(w**2 + h**2), area, -np.sum((mask[y:y+h, x:x+w] / 255) * np.log2(mask[y:y+h, x:x+w] / 255)), mean_color[0], mean_color[1], mean_color[2], median_color[0], median_color[1], median_color[2]])

# Verileri DataFrame'e dönüştürün
df = pd.DataFrame(data, columns=["No", "Center", "Width", "Height", "Diagonal", "Area", "Entropy", "Mean_R", "Mean_G", "Mean_B", "Median_R", "Median_G", "Median_B"])

# DataFrame'i Excel dosyasına yazdırın
df.to_excel("output.xlsx", index=False)

