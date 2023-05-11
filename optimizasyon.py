#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Örnek veri seti ve hedef fonksiyon
import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([8.5, 11, 13.5, 16])

# Model parametreleri
w = 0.5  # Ağırlık
b = 1.0  # Bias

# Hiperparametreler
eta = 0.01  # Öğrenme hızı
epochs = 100  # Epoch sayısı

# Gradient Descent algoritması
for i in range(epochs):
    y_pred = w * x + b  # Tahmin değerleri
    error = y_pred - y  # Hata
    w = w - eta * np.dot(error, x) / len(x)  # Ağırlık güncelleme
    b = b - eta * np.sum(error) / len(x)  # Bias güncelleme
    loss = np.mean(np.square(error))  # Loss hesaplama
    print(f"Epoch {i+1} - Loss: {loss:.4f}")


# In[2]:


# Örnek veri seti ve hedef fonksiyon
import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([8.5, 11, 13.5, 16])

# Model parametreleri
w = 0.5  # Ağırlık
b = 1.0  # Bias

# Hiperparametreler
eta = 0.01  # Öğrenme hızı
gamma = 0.9  # Momentum katsayısı
epochs = 100  # Epoch sayısı

# Momentum algoritması
vw = 0  # Ağırlık momentumu
vb = 0  # Bias momentumu
for i in range(epochs):
    y_pred = w * x + b  # Tahmin değerleri
    error = y_pred - y  # Hata
    vw = gamma * vw + eta * np.dot(error, x) / len(x)  # Ağırlık momentumu güncelleme
    vb = gamma * vb + eta * np.sum(error) / len(x)  # Bias momentumu güncelleme
    w = w - vw  # Ağırlık güncelleme
    b = b - vb  # Bias güncelleme
    loss = np.mean(np.square(error))  # Loss hesaplama
    print(f"Epoch {i+1} - Loss: {loss:.4f}")


# In[3]:


# Örnek veri seti ve hedef fonksiyon
import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([8.5, 11, 13.5, 16])

# Model parametreleri
w = 0.5  # Ağırlık
b = 1.0  # Bias

# Hiperparametreler
eta = 0.01  # Öğrenme hızı
gamma = 0.9  # Momentum katsayısı
epochs = 100  # Epoch sayısı

# Nesterov Accelerated Gradient algoritması
vw = 0  # Ağırlık momentumu
vb = 0  # Bias momentumu
for i in range(epochs):
    y_pred = w * x + b  # Tahmin değerleri
    w_ahead = w - gamma * vw  # Önceden hesaplanmış ağırlık
    y_pred_ahead = w_ahead * x + b  # Önceden hesaplanmış tahmin değerleri
    error = y_pred_ahead - y  # Hata
    vw = gamma * vw + eta * np.dot(error, x) / len(x)  # Ağırlık momentumu güncelleme
    vb = gamma * vb + eta * np.sum(error) / len(x)  # Bias momentumu güncelleme
    w = w - vw  # Ağırlık güncelleme
    b = b - vb  # Bias güncelleme
    loss = np.mean(np.square(error))  # Loss hesaplama
    print(f"Epoch {i+1} - Loss: {loss:.4f}")


# In[4]:


# Örnek veri seti ve hedef fonksiyon
import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([8.5, 11, 13.5, 16])

# Model parametreleri
w = 0.5  # Ağırlık
b = 1.0  # Bias

# Hiperparametreler
eta = 0.01  # Öğrenme hızı
eps = 1e-8  # Küçük bir sayı
epochs = 100  # Epoch sayısı

# Adagrad algoritması
sw = 0  # Ağırlık kare kökü momentumu
sb = 0  # Bias kare kökü momentumu
for i in range(epochs):
    y_pred = w * x + b  # Tahmin değerleri
    error = y_pred - y  # Hata
    grad_w = np.dot(error, x) / len(x)  # Ağırlık gradyanı
    grad_b = np.sum(error) / len(x)  # Bias gradyanı
    sw += grad_w ** 2  # Ağırlık kare kökü momentumu güncelleme
    sb += grad_b ** 2  # Bias kare kökü momentumu güncelleme
    w = w - eta * grad_w / (np.sqrt(sw) + eps)  # Ağırlık güncelleme
    b = b - eta * grad_b / (np.sqrt(sb) + eps)  # Bias güncelleme
    loss = np.mean(np.square(error))  # Loss hesaplama
    print(f"Epoch {i+1} - Loss: {loss:.4f}")


# In[5]:


# Örnek veri seti ve hedef fonksiyon
import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([8.5, 11, 13.5, 16])

# Model parametreleri
w = 0.5  # Ağırlık
b = 1.0  # Bias

# Hiperparametreler
rho = 0.95  # Ağırlık kare kökü momentumu katsayısı
eps = 1e-6  # Küçük bir sayı
epochs = 100  # Epoch sayısı

# Adadelta algoritması
sw = 0  # Ağırlık kare kökü momentumu
sb = 0  # Bias kare kökü momentumu
delta_w = 0  # Ağırlık değişim momentumu
delta_b = 0  # Bias değişim momentumu
for i in range(epochs):
    y_pred = w * x + b  # Tahmin değerleri
    error = y_pred - y  # Hata
    grad_w = np.dot(error, x) / len(x)  # Ağırlık gradyanı
    grad_b = np.sum(error) / len(x)  # Bias gradyanı
    sw = rho * sw + (1 - rho) * grad_w ** 2  # Ağırlık kare kökü momentumu güncelleme
    sb = rho * sb + (1 - rho) * grad_b ** 2  # Bias kare kökü momentumu güncelleme
    dw = - np.sqrt(delta_w + eps) / np.sqrt(sw + eps) * grad_w  # Ağırlık değişim momentumu hesaplama
    db = - np.sqrt(delta_b + eps) / np.sqrt(sb + eps) * grad_b  # Bias değişim momentumu hesaplama
    w = w + dw  # Ağırlık güncelleme
    b = b + db  # Bias güncelleme
    delta_w = rho * delta_w + (1 - rho) * dw ** 2  # Ağırlık değişim momentumu güncelleme
    delta_b = rho * delta_b + (1 - rho) * db ** 2  # Bias değişim momentumu güncelleme
    loss = np.mean(np.square(error))  # Loss hesaplama
    print(f"Epoch {i+1} - Loss: {loss:.4f}")


# In[6]:


# Örnek veri seti ve hedef fonksiyon
import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([8.5, 11, 13.5, 16])

# Model parametreleri
w = 0.5  # Ağırlık
b = 1.0  # Bias

# Hiperparametreler
eta = 0.01  # Öğrenme hızı
rho = 0.9  # Ağırlık kare kökü momentumu katsayısı
eps = 1e-6  # Küçük bir sayı
epochs = 100  # Epoch sayısı

# RMSProp algoritması
sw = 0  # Ağırlık kare kökü momentumu
sb = 0  # Bias kare kökü momentumu
for i in range(epochs):
    y_pred = w * x + b  # Tahmin değerleri
    error = y_pred - y  # Hata
    grad_w = np.dot(error, x) / len(x)  # Ağırlık gradyanı
    grad_b = np.sum(error) / len(x)  # Bias gradyanı
    sw = rho * sw + (1 - rho) * grad_w ** 2  # Ağırlık kare kökü momentumu güncelleme
    sb = rho * sb + (1 - rho) * grad_b ** 2  # Bias kare kökü momentumu güncelleme
    w = w - eta * grad_w / (np.sqrt(sw) + eps)  # Ağırlık güncelleme
    b = b - eta * grad_b / (np.sqrt(sb) + eps)  # Bias güncelleme
    loss = np.mean(np.square(error))  # Loss hesaplama
    print(f"Epoch {i+1} - Loss: {loss:.4f}")


# In[7]:


# Örnek veri seti ve hedef fonksiyon
import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([8.5, 11, 13.5, 16])

# Model parametreleri
w = 0.5  # Ağırlık
b = 1.0  # Bias

# Hiperparametreler
eta = 0.01  # Öğrenme hızı
beta1 = 0.9  # İlk moment momentumu katsayısı
beta2 = 0.999  # İkinci moment momentumu katsayısı
eps = 1e-8  # Küçük bir sayı
epochs = 100  # Epoch sayısı

# Adam algoritması
m_w = 0  # Ağırlık birinci moment momentumu
m_b = 0  # Bias birinci moment momentumu
v_w = 0  # Ağırlık ikinci moment momentumu
v_b = 0  # Bias ikinci moment momentumu
for i in range(epochs):
    y_pred = w * x + b  # Tahmin değerleri
    error = y_pred - y  # Hata
    grad_w = np.dot(error, x) / len(x)  # Ağırlık gradyanı
    grad_b = np.sum(error) / len(x)  # Bias gradyanı
    m_w = beta1 * m_w + (1 - beta1) * grad_w  # Ağırlık birinci moment momentumu güncelleme
    m_b = beta1 * m_b + (1 - beta1) * grad_b  # Bias birinci moment momentumu güncelleme
    v_w = beta2 * v_w + (1 - beta2) * grad_w ** 2  # Ağırlık ikinci moment momentumu güncelleme
    v_b = beta2 * v_b + (1 - beta2) * grad_b ** 2  # Bias ikinci moment momentumu güncelleme
    m_w_hat = m_w / (1 - beta1 ** (i+1))  # Ağırlık birinci moment momentumu düzeltme
    m_b_hat = m_b / (1 - beta1 ** (i+1))  # Bias birinci moment momentumu düzeltme
    v_w_hat = v_w / (1 - beta2 ** (i+1))  # Ağırlık ikinci moment momentumu düzeltme
    v_b_hat = v_b / (1 - beta2 ** (i+1))  # Bias ikinci moment momentumu düzeltme
    w = w - eta * m_w_hat / (np.sqrt(v_w_hat) + eps)  # Ağırlık güncelleme
    b = b - eta * m_b_hat / (np.sqrt(v_b_hat) + eps)  # Bias güncelleme
    loss = np.mean(np.square(error))  # Loss hesaplama
    print(f"Epoch {i+1} - Loss: {loss:.4f}")


# In[8]:


#Learning Rate Decay yöntemi 
# Örnek veri seti ve hedef fonksiyon
import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([8.5, 11, 13.5, 16])

# Model parametreleri
w = 0.5  # Ağırlık
b = 1.0  # Bias

# Hiperparametreler
eta = 0.1  # Başlangıç öğrenme hızı
epochs = 100  # Epoch sayısı
decay_rate = 0.1  # Öğrenme hızı azaltma oranı
decay_steps = 10  # Öğrenme hızı azaltma adım sayısı

# Learning Rate Decay algoritması
for i in range(epochs):
    y_pred = w * x + b  # Tahmin değerleri
    error = y_pred - y  # Hata
    grad_w = np.dot(error, x) / len(x)  # Ağırlık gradyanı
    grad_b = np.sum(error) / len(x)  # Bias gradyanı
    w = w - eta * grad_w  # Ağırlık güncelleme
    b = b - eta * grad_b  # Bias güncelleme
    loss = np.mean(np.square(error))  # Loss hesaplama
    if (i+1) % decay_steps == 0:  # Öğrenme hızı azaltma adımlarını kontrol etme
        eta = eta * decay_rate  # Öğrenme hızı azaltma
    print(f"Epoch {i+1} - Loss: {loss:.4f} - Learning Rate: {eta:.6f}")


# In[9]:


#AdaGrad
# Örnek veri seti ve hedef fonksiyon
import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([8.5, 11, 13.5, 16])

# Model parametreleri
w = 0.5  # Ağırlık
b = 1.0  # Bias

# Hiperparametreler
eta = 0.1  # Başlangıç öğrenme hızı
epochs = 100  # Epoch sayısı
epsilon = 1e-8  # Küçük bir sayı (sıfıra bölme hatasını önlemek için)

# AdaGrad algoritması
grad_sum_w = 0.0  # Gradyan toplamları (ağırlık)
grad_sum_b = 0.0  # Gradyan toplamları (bias)

for i in range(epochs):
    y_pred = w * x + b  # Tahmin değerleri
    error = y_pred - y  # Hata
    grad_w = np.dot(error, x) / len(x)  # Ağırlık gradyanı
    grad_b = np.sum(error) / len(x)  # Bias gradyanı
    grad_sum_w += np.square(grad_w)  # Gradyan toplamları (ağırlık)
    grad_sum_b += np.square(grad_b)  # Gradyan toplamları (bias)
    ada_grad_w = eta / np.sqrt(grad_sum_w + epsilon) * grad_w  # AdaGrad güncelleme (ağırlık)
    ada_grad_b = eta / np.sqrt(grad_sum_b + epsilon) * grad_b  # AdaGrad güncelleme (bias)
    w = w - ada_grad_w  # Ağırlık güncelleme
    b = b - ada_grad_b  # Bias güncelleme
    loss = np.mean(np.square(error))  # Loss hesaplama
    print(f"Epoch {i+1} - Loss: {loss:.4f} - Learning Rate: {eta:.6f}")


# In[10]:


#RMSProp
# Örnek veri seti ve hedef fonksiyon
import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([8.5, 11, 13.5, 16])

# Model parametreleri
w = 0.5  # Ağırlık
b = 1.0  # Bias

# Hiperparametreler
eta = 0.1  # Başlangıç öğrenme hızı
epochs = 100  # Epoch sayısı
beta = 0.9  # Hareketli ortalama faktörü

# RMSProp algoritması
grad_square_sum_w = 0.0  # Gradyan karelerinin hareketli ortalamaları (ağırlık)
grad_square_sum_b = 0.0  # Gradyan karelerinin hareketli ortalamaları (bias)

for i in range(epochs):
    y_pred = w * x + b  # Tahmin değerleri
    error = y_pred - y  # Hata
    grad_w = np.dot(error, x) / len(x)  # Ağırlık gradyanı
    grad_b = np.sum(error) / len(x)  # Bias gradyanı
    grad_square_sum_w = beta * grad_square_sum_w + (1 - beta) * np.square(grad_w)  # Gradyan karelerinin hareketli ortalamaları (ağırlık)
    grad_square_sum_b = beta * grad_square_sum_b + (1 - beta) * np.square(grad_b)  # Gradyan karelerinin hareketli ortalamaları (bias)
    rmsprop_w = eta / np.sqrt(grad_square_sum_w + 1e-8) * grad_w  # RMSProp güncelleme (ağırlık)
    rmsprop_b = eta / np.sqrt(grad_square_sum_b + 1e-8) * grad_b  # RMSProp güncelleme (bias)
    w = w - rmsprop_w  # Ağırlık güncelleme
    b = b - rmsprop_b  # Bias güncelleme
    loss = np.mean(np.square(error))  # Loss hesaplama
    print(f"Epoch {i+1} - Loss: {loss:.4f} - Learning Rate: {eta:.6f}")


# In[ ]:




