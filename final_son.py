#!/usr/bin/env python
# coding: utf-8

# In[18]:


import cv2
  
# read the image file
img = cv2.imread('finger_print.png', 2)
  
ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
  
# converting to its binary form
bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

cv2.imwrite('binary_image.png', bw_img)
  
cv2.imshow("Binary", bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[20]:


import cv2

# resmi oku
image = cv2.imread('binary_image.png')

# resmi üzerine erozyon uygula
eroded = cv2.erode(image, kernel=None, iterations=1)

# resmi 'eroded_image.png' olarak kaydet
cv2.imwrite('eroded_image.png', eroded)

# erozyon uygulanmış resmi ekranda göster
cv2.imshow('Eroded Image', eroded)

# ekranda gösterilen resmi beklet
cv2.waitKey(0)

# tüm pencereleri kapat
cv2.destroyAllWindows()


# In[21]:


import cv2
import numpy as np

import cv2

# resmi oku
image = cv2.imread('binary_image.png')

# resmi üzerine açma işlemini uygula
opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel=None)

# resmi 'opened.png' olarak kaydet
cv2.imwrite('opened.png', opened)

# açma işlemini gerçekleştirilmiş resmi ekranda göster
cv2.imshow('Opened Image', opened)

# ekranda gösterilen resmi beklet
cv2.waitKey(0)

# tüm pencereleri kapat
cv2.destroyAllWindows()


# In[23]:


import cv2
import numpy as np

import cv2

# resmi oku
image = cv2.imread('binary_image.png')

# resmi üzerine açma işlemini uygula
opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel=None)

# opened üzerine resmi üzerine kapatma işlemini uygula
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel=None)

# resmi 'closed.png' olarak kaydet
cv2.imwrite('closed.png', closed)

# closed işlemini gerçekleştirilmiş resmi ekranda göster
cv2.imshow('Closed Image', closed)

# ekranda gösterilen resmi beklet
cv2.waitKey(0)

# tüm pencereleri kapat
cv2.destroyAllWindows()


# In[ ]:




