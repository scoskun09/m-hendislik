#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Problemin tanımı (20)
Problemin önemi (20)
Veri seti hakkında istatistik (20)
Zaman yönetimi (20)
Diğer sorular (20)
https://lhncbc.nlm.nih.gov/LHC-downloads/dataset.html sitesinden download kısmına girdim

Montgomery County CXR Seti : Bu veri setindeki görüntüler ABD, Montgomery County, Sağlık ve İnsan Hizmetleri Departmanının TB Kontrol Programından alınmıştır. Bu set, 80'i normal ve 58'i anormal olan 138 posterior-anterior CXR içerir ve tezahürleri TB ile tutarlıdır. Tüm görüntülerin kimliği gizlenmiştir ve PNG formatında sol ve sağ PA-görünümlü akciğer maskeleriyle birlikte mevcuttur. Veri seti ayrıca, 1024 × 1024 yeniden boyutlandırılmış görüntüler ve radyoloji okumaları için iki radyologdan gelen konsensüs ek açıklamalarını içerir. Montgomery County CXR Setini İndirin
veri setini yükle ve göster işlemi yaptım 


# In[12]:


import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Veri kümesini yükle
df = pd.read_csv("montgomery_consensus_roi.csv")
#veri kümesini göster
print(df)


# In[ ]:


Bu kod bloğunda, pandas `read_csv()` fonksiyonu kullanılarak "montgomery_consensus_roi.csv" dosyası yükleniyor. Daha sonra, OpenCV kullanılarak görüntüler ve etiketler listeleri oluşturuluyor. Veriler, `train_test_split()` fonksiyonu kullanılarak eğitim ve test setlerine bölünüyor. Test setinin boyutu %20 olduğu için `test_size` parametresi 0.2 olarak belirtiliyor. Son olarak, eğitim ve test setlerinin boyutları yazdırılıyor.

Bu değişiklikleri yaparak, kodunuzu sütun adlarına göre değiştirebilirsiniz.


# In[14]:


import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Veri kümesini yükle
df = pd.read_csv("montgomery_consensus_roi.csv")

# Görüntüleri yükleyin ve sınıf etiketlerini ayarlayın
images = []
labels = []
for i in range(len(df)):
    img = cv2.imread(df.iloc[i]['patientId'], cv2.IMREAD_GRAYSCALE)
    images.append(img)
    labels.append(df.iloc[i]['Labels'])

# Verileri eğitim ve test setlerine bölün
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Eğitim ve test setlerinin boyutlarını yazdırın
print("Training data size:", len(X_train))
print("Testing data size:", len(X_test))


# In[ ]:


Bu kod, "montgomery_consensus_roi.csv" adlı bir veri setindeki özellikleri ve kategorileri tanımlar. Veri setindeki "Labels" sütunundaki değerleri dummies sütunlarına dönüştürür ve kategori sütunlarını veri setine ekler.

Bu örnek, özellikleri ve kategorileri tanımlamak ve dummies sütunlarına dönüştürmek için basit bir yöntemdir. Veri kümesinin özelliklerine ve kategorilerine göre farklı yöntemler kullanmak mümkündür.


# In[18]:


import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Veri setini yükle
df = pd.read_csv("montgomery_consensus_roi.csv")

# Özellikleri tanımla
ozellikler = ["x_dis", "y_dis", "width_dis", "height_dis", "Labels"]

# Kategorilere ayır
label_kategorileri = pd.get_dummies(df["Labels"])

# Kategori sütunlarını veri setine ekle
df = pd.concat([df, label_kategorileri], axis=1)
print(df)


# In[ ]:




