# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 07:09:04 2019

@author: Hamid.t
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


from sklearn.datasets import load_breast_cancer


data = load_breast_cancer()

df  = pd.DataFrame(data.data)

df.columns= data.feature_names
df['target_label']=data.target

os.makedirs('plots/seaborn_heatmap', exist_ok=True)

sns.set()

fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(df.corr(), annot=True, cmap='autumn')
ax.set_xticklabels(df.columns, rotation=45)
ax.set_yticklabels(df.columns, rotation=45)
plt.savefig('plots/seaborn_heatmap/cancer_heatmap.png')

plt.close()