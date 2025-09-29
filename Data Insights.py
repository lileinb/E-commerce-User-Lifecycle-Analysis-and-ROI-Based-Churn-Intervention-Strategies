#--- 数据加载和洞察 ---
import pandas as pd
import numpy as np
df_raw = pd.read_csv('D:\Onediver\OneDrive\Desktop\E-commerce User Lifecycle Analysis and ROI-Based Churn Intervention Strategies\Online Retail Dataset\online_retail.csv',encoding='latin1')
print(f"原始数据维度：{df_raw.shape[0]}行，{df_raw.shape[1]}列")
print(f"原始数据列名：{df_raw.columns.tolist()}")
df_raw.info()
print(df_raw.head())
print(df_raw.describe())

missing_count = df_raw.isnull().sum()
print(f"缺失值数量：{missing_count}")
print(f"缺失值比例：{missing_count/df_raw.shape[0]}")
print(f"缺失值比例：{missing_count/df_raw.shape[0]}")





