#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Soru-1 /Seda Coşkun Eliküçük/Mühendislik Tamamlama/220304131

#Mükemmel Sayı Kontrolü

def mukemmelsayimi(sayi):
    toplam=0
    for i in range(1,sayi):
        if sayi % i == 0:
            toplam=toplam+i
    if toplam==sayi:
        return True
    return False

def mukemmelsayilaribul(sayi):
    sayilar=""
    for i in range(1,sayi):
        if mukemmelsayimi(i):
            sayilar=(sayilar+","+str(i))
    return sayilar.strip(",")

print(mukemmelsayilaribul(500))


# In[5]:


#Soru-2 

#Öğrenci Numarasının mükemmel sayı olup olmadığının kontrolü 
def mukemmelsayimi(sayi):
    toplam=0
    for i in range(1,sayi):
        if sayi % i == 0:
            toplam=toplam+i
    if toplam==sayi:
        return True
    return False
if(mukemmelsayimi(220304131)):
   print("Mükemmel Sayıdır")
else:
   print("Mükemmel Sayı Değildir")


# In[6]:


#Soru-3 
#Dosya Uzantısı Bulma

def dosyaadi(ad):
    isim=ad.split(".")
    return isim[len(isim)-1]
oku=input("dosya ismi")
print(dosyaadi(oku))


# In[7]:


#Soru-4
#a şıkkı
ogrenci={
    "isim":"Seda",
    "yas":"35",
    "dersler":"Matematik"
}
print (ogrenci) #b şıkkı
ogrenci["okul"]="siirt ünv" #c şıkkı
ogrenci.update({"isim":"mehmet","yas":"25"}) #d şıkkı
print(ogrenci)
ogrenci.pop("yas")#e şıkkı
print(ogrenci)
print(len(ogrenci.keys()))#f şıkkı
print(ogrenci.keys())#g şıkkı
for k,v in ogrenci.items():
    print(k,v)


# In[8]:


#Soru-5

def kuplertoplam2(n):
    for i1 in range(1, n+1):
        sa_str = str(i1)
        bs = len(sa_str)
        sd = []
        for j1 in range(1, bs+1):
            sd.append(int(sa_str[j1-1]))
        top = sum([x**3 for x in sd])  # Küp toplamı hesaplaması güncellendi
        if i1 == top:
            print(i1)
kuplertoplam2(153)


# In[ ]:




