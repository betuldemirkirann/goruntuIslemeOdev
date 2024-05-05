import cv2
import numpy as np
from matplotlib import pyplot as plt

bgr_img = cv2.imread('eyes.jpg')

if len(bgr_img.shape) == 3:  # Eğer renkli bir görüntüyse 
    b,g,r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r,g,b])
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
else:  #(siyah-beyaz)
    gray_img = bgr_img #renk değişmesin

img = cv2.medianBlur(gray_img, 35)

# Orijinal renkli görüntünün aynısını oluştur
cimg = rgb_img.copy()

# Hough dönüşümü uygulama
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                            param1=50, param2=30, minRadius=0, maxRadius=0)

if circles is not None:  # Daireler bulunduysa
    circles = np.uint16(np.around(circles))
    
    for i in circles[0,:]:
        # Dairenin dış çizgisi
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Dairenin merkezi
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
#cimg değişkeni orijinal renkli görüntüden oluşturulmuş bir kopya olarak tanımlanır. Daha sonra, Hough dönüşümü uygulanmış gri tonlamalı görüntüye çizilen daireler bu kopya görüntüye eklenir.
#Bu sayede, Hough dönüşümü uygulanmış görüntü, orijinal renkli görüntü ile aynı yapıya sahip olur.

plt.subplot(121), plt.imshow(rgb_img)
plt.title('Orijinal Görüntü'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cimg)
plt.title('Hough Dönüşümü'), plt.xticks([]), plt.yticks([])
plt.show()
