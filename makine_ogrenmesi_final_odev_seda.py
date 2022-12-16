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


# In[ ]:




