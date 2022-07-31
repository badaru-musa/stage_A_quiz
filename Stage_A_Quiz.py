#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np


# In[4]:


url = 'https://github.com/HamoyeHQ/HDSC-Introduction-to-Python-for-machine-learning/files/7768140/FoodBalanceSheets_E_Africa_NOFLAG.csv'
df = pd.read_csv(url, encoding='latin-1')
display(df.head())
display(df.info())


# In[6]:


q10 = df.groupby("Item")[['Y2014', 'Y2017']].sum()
display(q11.loc['Animal fats'])


# In[7]:


q11 = df.Y2015.describe()
display(q12)


# In[13]:


q12 = df['Y2016'].isnull().sum()
display(q13)
q13 = df['Y2016'].value_counts(normalize=True,dropna=False)
display(q14)


# In[17]:


q14 = df.corr().nlargest(2, 'Element Code')['Element Code']
display(q15)


# In[23]:


q15 = df.groupby("Element").sum().loc['Import Quantity', 'Y2014': 'Y2018']
display(q16)


# In[26]:


q16 = df.groupby("Element")[['Y2014']].sum().loc['Production']
display(q17)


# In[27]:


v = df[['Y2018', 'Element']].groupby("Element").sum()
q17 = v.nlargest(1, 'Y2018')
display(q17)
q18 = v.nsmallest(3, 'Y2018')
display(q18)


# In[29]:


q19 = df.groupby(['Area', 'Element'])['Y2018'].sum()
display(q19.loc[('Algeria', 'Import Quantity')])


# In[31]:


print(df['Area'].nunique())


# In[ ]:




