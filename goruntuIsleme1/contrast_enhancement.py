import cv2
import matplotlib.pyplot as plt
import numpy as np

def standard_sigmoid(x, alpha=5, beta=0.5):
    """
    Standart sigmoid fonksiyonu
    """
    return 1 / (1 + np.exp(-alpha * (x - beta)))

def shifted_sigmoid(x, alpha=10, beta=0.5, gamma=0.5):
    """
    Yatay kaydırılmış sigmoid fonksiyonu
    """
    return 1 / (1 + np.exp(-alpha * (x - beta + gamma)))

def tilted_sigmoid(x, alpha=5, beta=0.5, theta=2):
    """
    Eğimli sigmoid fonksiyonu
    """
    return 1 / (1 + np.exp(-alpha * (x - beta))) ** theta

def custom_function(x):
    """
    Kullanıcı tarafından üretilen özel fonksiyon
    """
    # Örnek bir işlev: Kare alma
    return x ** 2

# Görüntü dosyasının adı
image_path = "image.jpg"

# Görüntüyü yükleme (renkli olarak)
image = cv2.imread(image_path)

# Görüntüyü grayscale olarak dönüştürme
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Görüntüyü [0, 1] aralığına normalize etme
normalized_image = grayscale_image / 255.0

# Standart sigmoid fonksiyonunu uygulama
enhanced_image_standard_sigmoid = standard_sigmoid(normalized_image)

# Yatay kaydırılmış sigmoid fonksiyonunu uygulama
enhanced_image_shifted_sigmoid = shifted_sigmoid(normalized_image, gamma=0.5)  # Örnek bir gamma değeri

# Eğimli sigmoid fonksiyonunu uygulama
enhanced_image_tilted_sigmoid = tilted_sigmoid(normalized_image, theta=2)  # Örnek bir theta değeri

# Kullanıcı tarafından üretilen özel fonksiyonu uygulama
enhanced_image_custom_function = custom_function(normalized_image)

# S-Curve yöntemi ile kontrast güçlendirme
enhanced_image_s_curve = enhanced_image_standard_sigmoid * enhanced_image_shifted_sigmoid * enhanced_image_tilted_sigmoid * enhanced_image_custom_function

# Orijinal görüntüyü ve S-Curve yöntemi ile işlenmiş görüntüyü yatay olarak gösterme
plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)
plt.imshow(normalized_image, cmap='gray')
plt.axis('off')
plt.title("Orijinal Görüntü")

plt.subplot(1, 2, 2)
plt.imshow(enhanced_image_s_curve, cmap='gray')
plt.axis('off')
plt.title("S-Curve Yöntemi ile Güçlü Kontrast Güçlendirme")

plt.tight_layout()
plt.show()
