#!/usr/bin/env python
# coding: utf-8

# In[25]:


import cv2

# resmi oku
image = cv2.imread('finger_print.png')

# resmi gri seviyesine dönüştür
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# resmi üzerine erozyon uygula
eroded = cv2.erode(gray, kernel=None, iterations=1)

# resmi 'eroded_image.png' olarak kaydet
cv2.imwrite('eroded_image.png', eroded)

# erozyon uygulanmış resmi ekranda göster
cv2.imshow('Eroded Image', eroded)

# ekranda gösterilen resmi beklet
cv2.waitKey(0)

# tüm pencereleri kapat
cv2.destroyAllWindows()


# In[26]:


import cv2
import numpy as np

# resmi oku
image = cv2.imread('finger_print.png')

# resmi gri seviyesine dönüştür
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# resmi üzerine açma işlemini uygula
opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel=None)

# resmi 'opened.png' olarak kaydet
cv2.imwrite('opened.png', opened)

# açma işlemini gerçekleştirilmiş resmi ekranda göster
cv2.imshow('Opened Image', opened)

# ekranda gösterilen resmi beklet
cv2.waitKey(0)

# tüm pencereleri kapat
cv2.destroyAllWindows()


# In[30]:


import cv2

# resmi oku
image = cv2.imread('finger_print.png')

# resmi gri seviyesine dönüştür
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# resmi üzerine genişleme işlemini uygula
dilated = cv2.dilate(gray, kernel=None, iterations=1)

# resmi 'dilated.png' olarak kaydet
cv2.imwrite('dilated.png', dilated)

# resmi aç
cv2.imshow('Dilated Image', dilated)

# ekranda gösterilen resmi beklet
cv2.waitKey(0)

# tüm pencereleri kapat
cv2.destroyAllWindows()


# In[33]:


import cv2
import numpy as np

# resmi oku
image = cv2.imread('dilated.png')

# resmi gri seviyesine dönüştür
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# resmi üzerine açma işlemini uygula
closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel=None)

# resmi 'opened.png' olarak kaydet
cv2.imwrite('closed.png', closed)

# açma işlemini gerçekleştirilmiş resmi ekranda göster
cv2.imshow('Closed Image', closed)

# ekranda gösterilen resmi beklet
cv2.waitKey(0)

# tüm pencereleri kapat
cv2.destroyAllWindows()


# In[ ]:




