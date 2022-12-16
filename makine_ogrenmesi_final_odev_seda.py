#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas an pd
import numpy as np
import seaborn as sns


# In[3]:


from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# In[1]:


from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


iris=load_iris()
data=iris.data
feature_names=iris.feature_names
y=iris.target
y


# In[3]:


df=pd.DataFrame(data,columns=feature_names)
df["sinif"]=y
x=data
df.head()


# In[5]:


from sklearn.decomposition import PCA
pca=PCA(n_components=2,whiten=True)
pca.fit(x)
x_pca=pca.transform(x)
print("variance ratio:",pca.explained_variance_ratio_)
print("sum:",sum(pca.explained_variance_ratio_))


# In[8]:


from sklearn.decomposition import PCA
pca=PCA(n_components=2,whiten=True)
pca=PCA(whiten=True).fit(x)
plt.plot(np.cumsum(pca.explained_variance_ratio))
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
plt.show()


# In[10]:


pca=PCA(whiten=True).fit(x)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
plt.show()


# In[11]:


df_sns=pd.DataFrame({"var":pca.explained_variance_ratio_,"PC":["PC1","PC2","PC3","PC4"]})
df_sns


# In[12]:


df_sns=pd.DataFrame({"var":pca.explained_variance_ratio_,"PC":["PC1","PC2","PC3","PC4"]})
df_sns


# In[13]:


sns.barplot(x="PC",y="var",data=df_sns,color="c")
plt.ylabel("Variance Explained")
plt.xlabel("Principle Components")
plt.show()


# In[16]:


df["p1"]=x_pca[:,0]
df["p2"]=x_pca[:,1]
color=["red","green","blue"]


# In[18]:


for each in range(3):
 plt.scatter(df.p1[df.sinif==each],df.p2[df.sinif==each],color=color[each],label=iris.target_names[each])
plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()


# In[ ]:




