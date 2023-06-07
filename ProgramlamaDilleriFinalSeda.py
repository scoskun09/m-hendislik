#!/usr/bin/env python
# coding: utf-8

# In[20]:


import math

class Nokta:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
class Dogru:
    def __init__(self, nokta1, nokta2):
        self.nokta1 = nokta1
        self.nokta2 = nokta2
        
    def kesisme_noktasi(self, diger_dogru):
        x1, y1 = self.nokta1.x, self.nokta1.y
        x2, y2 = self.nokta2.x, self.nokta2.y
        x3, y3 = diger_dogru.nokta1.x, diger_dogru.nokta1.y
        x4, y4 = diger_dogru.nokta2.x, diger_dogru.nokta2.y
        
        # Doğruların eğimleri
        ma = (y2 - y1) / (x2 - x1)
        mb = (y4 - y3) / (x4 - x3)
        
        # Eğimlerin farkı
        m_farki = ma - mb
        
        # Doğruların kesişim noktasının x koordinatı
        x_kesisim = (ma * x1 - mb * x3 + y3 - y1) / m_farki
        
        # Doğruların kesişim noktasının y koordinatı
        y_kesisim = ma * (x_kesisim - x1) + y1
        
        return Nokta(x_kesisim, y_kesisim)
    
class Cember:
    def __init__(self, nokta1, nokta2, nokta3):
        self.nokta1 = nokta1
        self.nokta2 = nokta2
        self.nokta3 = nokta3
        
    def merkez_ve_yaricap(self):
        dogru1 = Dogru(self.nokta1, self.nokta2)
        dogru2 = Dogru(self.nokta2, self.nokta3)
        
        # Doğruların kesişim noktası
        merkez = dogru1.kesisme_noktasi(dogru2)
        
        # Merkez noktası ile 1. nokta arasındaki mesafe
        yaricap = math.sqrt((merkez.x - self.nokta1.x) ** 2 + (merkez.y - self.nokta1.y) ** 2)
        
        return merkez, yaricap
nokta1 = Nokta(6, 3)
nokta2 = Nokta(4, 1)
nokta3 = Nokta(2.59, 1.59)

cember = Cember(nokta1, nokta2, nokta3)
merkez, yaricap = cember.merkez_ve_yaricap()

print("Merkez Noktası: ({0:.2f}, {1:.2f})".format(merkez.x, merkez.y))
print("Yarıçap Uzunluğu: {0:.2f}".format(yaricap))


# In[ ]:




