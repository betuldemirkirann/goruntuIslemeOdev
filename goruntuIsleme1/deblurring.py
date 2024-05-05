import numpy as np
import cv2
import matplotlib.pyplot as plt

def deblur(image, kernel_size=(5, 5), sigma=0.5):
    # Gürültüyü azaltmak için görüntüyü önce bulanıklaştır
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)

    # Deblurring işlemi için Laplace filtresi kullanarak kenarları tespit et
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # Kenarları bulanıklaştırarak, diğer kısımlardaki gürültüyü azaltma
    laplacian_uint8 = np.clip((laplacian + 128), 0, 255).astype(np.uint8)

    deblurred = cv2.addWeighted(image, 1.5, laplacian_uint8, -0.5, 5)

    return deblurred

image = cv2.imread('input_image.jpg')

# Deblurring işlemi
deblurred_image = deblur(image, sigma=0.5)

# Görüntüleri göster
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Deblurred Image')
plt.imshow(cv2.cvtColor(deblurred_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
