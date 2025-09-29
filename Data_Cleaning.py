import pandas as pd
import numpy as np
df_raw = pd.read_csv(r'D:\Onediver\OneDrive\Desktop\E-commerce User Lifecycle Analysis and ROI-Based Churn Intervention Strategies\Online Retail Dataset\online_retail.csv', encoding='latin1')
df_cleaned = df_raw.copy()
print({len(df_cleaned)})
df_cleaned.dropna(subset=['CustomerID'],inplace=True)
print(f"处理CustomerID缺失值后，剩余行数: {len(df_cleaned)}")
df_cleaned = df_cleaned[df_cleaned['Quantity'] > 0]
print(f"处理退货数量，剩余行数: {len(df_cleaned)}")
df_cleaned = df_cleaned[df_cleaned['UnitPrice'] > 0]
print(f"处理保留有效零售商品价格，剩余行数: {len(df_cleaned)}")

# 步骤1: 定义截图中的已知数
rows_step1 = 406829  # 处理CustomerID缺失值后
rows_step2 = 397924  # 处理退货数量后
rows_step3 = 397884  # 处理零价格商品后
original_rows = 541909 # 原始数据行数

# 步骤2: 分析“处理退货数量”这一步
dropped_in_step2 = rows_step1 - rows_step2
percentage_dropped_step2 = (dropped_in_step2 / rows_step1) * 100

# 步骤3: 分析“处理零价格商品”这一步
dropped_in_step3 = rows_step2 - rows_step3
percentage_dropped_step3 = (dropped_in_step3 / rows_step2) * 100

# 步骤4: 计算总体数据保留率
total_rows_final = rows_step3
percentage_retained = (total_rows_final / original_rows) * 100

# 步骤5: 打印格式化的分析报告
print("--- 数据清洗全流程：效果分析报告 ---")
print(f"原始数据总行数: {original_rows}")
print("-" * 35)
print(f"第1步 [处理CustomerID缺失] 后剩余: {rows_step1} 行")
print(f"第2步 [处理退货订单] 后剩余: {rows_step2} 行")
print(f"  └— 本步过滤掉 {dropped_in_step2} 行 (占上一步数据的 {percentage_dropped_step2:.2f}%)")
print(f"第3步 [处理零价格商品] 后剩余: {rows_step3} 行")
print(f"  └— 本步过滤掉 {dropped_in_step3} 行 (占上一步数据的 {percentage_dropped_step3:.2f}%)")
print("-" * 35)
print(f"最终得到干净数据集，总行数: {total_rows_final}")
print(f"数据总体保留率: {percentage_retained:.2f}%")

# 步骤1: 转换 InvoiceDate 的数据类型
# 将 object (文本) 类型转换为 datetime (日期时间) 类型
df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])

# 步骤2: 转换 CustomerID 的数据类型
# 先从 float (浮点数) 转为 int (整数) 去掉小数点，再转为 str (字符串)
df_cleaned['CustomerID'] = df_cleaned['CustomerID'].astype(int).astype(str)

# 步骤3: 创建新的特征列 TotalPrice
# 向量化操作，将两列直接相乘
df_cleaned['TotalPrice'] = df_cleaned['Quantity'] * df_cleaned['UnitPrice']

# 步骤4: 验证转换和创建的结果
print("--- [数据转换与特征创建完成] ---")
print("\n--- 1. 检查数据类型是否已正确转换 (df_cleaned.info) ---")
df_cleaned.info()

print("\n--- 2. 预览包含TotalPrice的新数据集 (前5行) ---")
print(df_cleaned.head())
print(df_cleaned.columns.tolist())

snapshot_date = df_cleaned['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm_df = df_cleaned.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})
rfm_df.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'MonetaryValue'
}, inplace=True)
print("--- [RFM分析完成] ---")
print("\n--- 1. 检查RFM结果 (前5行) ---")
print(rfm_df.head())
print("\n--- 2. 检查RFM结果的统计信息 ---")
print(rfm_df.describe())

print("\n--- 3. 检查RFM结果的分布情况 ---")
print(rfm_df.hist(bins=20, figsize=(15, 10)))                                                                           


# 步骤1: 为 R(Recency) 打分 (值越小，得分越高)
r_labels = [4, 3, 2, 1]
rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], q=4, labels=r_labels, duplicates='drop')

# 步骤2: 为 F(Frequency) 和 M(MonetaryValue) 打分 (值越大，得分越高)
f_m_labels = [1, 2, 3, 4]
rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), q=4, labels=f_m_labels)
rfm_df['M_Score'] = pd.qcut(rfm_df['MonetaryValue'].rank(method='first'), q=4, labels=f_m_labels)
rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'].rank(method='first'), q=4, labels=r_labels)
# 步骤3: 预览打分后的结果
print("--- [RFM打分完成] ---")
print(rfm_df.head())



# 步骤1: 将独立的R, F, M分数合并成一个总的RFM分数
# 我们将数字分数先转成字符串，再拼接在一起
rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)


# 步骤2: 创建一个“分层字典”作为我们的“翻译地图”
# key是RFM分数，value是我们赋予的业务标签
seg_map = {
    r'[1-2][1-2]': '低价值客户',
    r'[1-2][3-4]': '中价值客户',
    r'3[1-2]': '潜力客户',
    r'4[1-2]': '新客户',
    r'33': '忠诚客户',
    r'34': '高价值忠诚客户',
    r'4[3-4]': '高价值新客户',
    r'44': '重要价值客户'
}

# 步骤3: 使用正则表达式和map函数，为每个RFM分数打上业务标签
# 这是一个更简洁和强大的分群方法
rfm_df['Segment'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str)
rfm_df['Segment'] = rfm_df['Segment'].replace(seg_map, regex=True)


# 步骤4: 将最终处理好的数据导出为CSV文件，供Tableau使用
rfm_df.to_csv('rfm_for_tableau.csv')


# 步骤5: 预览最终的数据表
print("--- [数据准备完成，可用于Tableau] ---")
print("\n--- 1. 最终RFM数据表预览 (前5行) ---")
print(rfm_df.head())
print(f"\n文件 'rfm_for_tableau.csv' 已保存到你当前的工作目录。")


# 步骤1: 计算每个用户的每次购买时间
# 我们需要一个新的DataFrame，其中每一行代表一个用户的一次独立购买（以天为单位）
df_purchase_dates = df_cleaned.groupby('CustomerID')['InvoiceDate'].apply(lambda x: x.dt.date.unique()).reset_index()
df_purchase_dates.columns = ['CustomerID', 'PurchaseDates']

# 步骤2: 计算每位用户的购买间隔 (inter-purchase interval)
# 我们计算一个用户相邻两次购买之间隔了多少天
purchase_intervals = []
for index, row in df_purchase_dates.iterrows():
    dates = sorted(row['PurchaseDates'])
    if len(dates) > 1: # 必须至少有两次购买才能计算间隔
        for i in range(len(dates) - 1):
            interval = (dates[i+1] - dates[i]).days
            purchase_intervals.append(interval)

# 步骤3: 使用分位数来确定流失阈值
# 我们计算所有购买间隔的95分位数，作为我们的“流失时间窗口”
# 这意味着，如果一个用户超过这段时间没来，他就很可能已经流失了
churn_threshold = np.percentile(purchase_intervals, 95)

print("--- [流失定义分析] ---")
print(f"根据所有用户的购买间隔分析，我们计算出的95分位数是: {churn_threshold:.0f} 天")
print(f"因此，我们将定义：如果一个用户超过 {churn_threshold:.0f} 天未购买，则视为'已流失'。")


# 步骤4: 在 rfm_df 中应用流失定义，创建目标变量
# Recency 代表用户最近一次购买距今的天数
# 如果这个天数大于我们的流失阈值，那么该用户就已流失 (Churn = 1)
rfm_df['Churn'] = rfm_df['Recency'].apply(lambda x: 1 if x > churn_threshold else 0)


# 步骤5: 查看结果
print("\n--- [在RFM表中添加Churn标签后预览] ---")
print(rfm_df.head())

print("\n--- [流失与未流失用户数量统计] ---")
print(rfm_df['Churn'].value_counts())


# 特征1: 平均客单价 (Average Order Value)
# 思路：客单价的变化可能反映用户消费能力或意愿的改变。
aov_df = df_cleaned.groupby('CustomerID')['TotalPrice'].sum() / df_cleaned.groupby('CustomerID')['InvoiceNo'].nunique()
aov_df = aov_df.reset_index()
aov_df.columns = ['CustomerID', 'AvgOrderValue']
# 将新特征合并到主RFM表中
rfm_df = pd.merge(rfm_df, aov_df, on='CustomerID')


# 特征2: 购买周期稳定性 (Standard Deviation of Purchase Intervals)
# 思路：购买周期越不稳定（标准差越大），说明用户的购买行为越无规律，可能是流失的前兆。
# (我们复用之前计算购买间隔的代码)
df_purchase_dates = df_cleaned.groupby('CustomerID')['InvoiceDate'].apply(lambda x: x.dt.date.unique()).reset_index()
df_purchase_dates.columns = ['CustomerID', 'PurchaseDates']

interval_std_list = []
for index, row in df_purchase_dates.iterrows():
    dates = sorted(row['PurchaseDates'])
    customer_id = row['CustomerID']
    if len(dates) > 1:
        intervals = []
        for i in range(len(dates) - 1):
            interval = (dates[i+1] - dates[i]).days
            intervals.append(interval)
        interval_std = np.std(intervals) if intervals else 0
        interval_std_list.append({'CustomerID': customer_id, 'PurchaseIntervalStd': interval_std})
    else: # 如果只有一次购买，则稳定性为0
        interval_std_list.append({'CustomerID': customer_id, 'PurchaseIntervalStd': 0})

interval_std_df = pd.DataFrame(interval_std_list)
# 将新特征合并到主RFM表中
rfm_df = pd.merge(rfm_df, interval_std_df, on='CustomerID')


# 特征3: 购买商品品类多样性 (Number of Unique StockCodes)
# 思路：购买的品类数越多，通常代表用户与平台的粘性越高。品类数减少可能是兴趣下降的信号。
uniqueness_df = df_cleaned.groupby('CustomerID')['StockCode'].nunique().reset_index()
uniqueness_df.columns = ['CustomerID', 'UniqueProductsCount']
# 将新特征合并到主RFM表中
rfm_df = pd.merge(rfm_df, uniqueness_df, on='CustomerID')


# 步骤4: 查看最终的特征工程结果
print("--- [特征工程完成] ---")
print("\n--- 1. 预览包含所有预测特征的最终数据集 (前5行) ---")
print(rfm_df.head())

print("\n--- 2. 检查新特征的数据类型与缺失值 ---")
rfm_df.info()

print("\n--- [正在保存最终处理结果到文件] ---")
rfm_df.to_csv('rfm_final_for_modeling.csv')
print(f"文件 'rfm_final_for_modeling.csv' 已成功保存。")
print("数据清洗和特征工程阶段完成！")


# 步骤1: 设定一个合理的假设利润率
# 这是一个业务假设，在实际项目中需要和业务方确认。这里我们假设综合利润率为25%。
PROFIT_MARGIN = 0.25

# 步骤2: 计算每个用户的历史利润贡献
# 我们在rfm_df上创建一个新列'Profit'
rfm_df['Profit'] = rfm_df['MonetaryValue'] * PROFIT_MARGIN

# 步骤3: 按Segment分组，聚合计算每个分层的核心LTV指标
ltv_df = rfm_df.groupby('Segment').agg(
    Customer_Count = ('Segment', 'size'),  # 计算每个分层的用户数
    Total_Profit = ('Profit', 'sum')       # 计算每个分层的总利润
)

# 步骤4: 计算每个分层的人均LTV (我们估算的LTV)
ltv_df['Avg_LTV_per_Customer'] = ltv_df['Total_Profit'] / ltv_df['Customer_Count']

# 步骤5: 按人均LTV降序排序，并展示结果
ltv_df_sorted = ltv_df.sort_values(by='Avg_LTV_per_Customer', ascending=False)

# 步骤6: 打印最终的LTV分析报告
print("\n--- [各用户分层LTV分析报告] ---")
print(ltv_df_sorted)


# ... (这里才是你原来保存文件的代码) ...
# print("\n--- [正在保存最终处理结果到文件] ---")
# rfm_df.to_csv('rfm_final_for_modeling.csv')
# print(f"文件 'rfm_final_for_modeling.csv' 已成功保存。")
