import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE 
import seaborn as sns
import matplotlib.pyplot as plt

# --- 核心代码 ---

# 步骤1: 加载数据并定义X和y
rfm_df = pd.read_csv('rfm_final_for_modeling.csv')
features = ['F_Score', 'M_Score', 'AvgOrderValue', 'PurchaseIntervalStd', 'UniqueProductsCount']
X = rfm_df[features]
y = rfm_df['Churn']

# 步骤2: 划分原始的训练集和测试集 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 步骤3: [核心优化] 应用SMOTE对训练数据进行过采样
print("--- [应用SMOTE前，训练集中Churn标签的分布] ---")
print(y_train.value_counts())

# 初始化SMOTE
smote = SMOTE(random_state=42)
# 使用.fit_resample()方法进行过采样
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\n--- [应用SMOTE后，新训练集中Churn标签的分布] ---")
print(y_train_smote.value_counts())


# 步骤4: 使用平衡后的新数据重新训练模型
print("\n--- [正在使用平衡后的数据重新训练模型...] ---")
log_reg_model_smote = LogisticRegression(random_state=42)
log_reg_model_smote.fit(X_train_smote, y_train_smote)
print("模型重新训练完成！")


# 步骤5: 在原始测试集上进行预测和评估
y_pred_smote = log_reg_model_smote.predict(X_test)

print("\n--- [优化后模型的性能评估报告] ---")
print(classification_report(y_test, y_pred_smote))

print("\n--- [优化后模型的混淆矩阵] ---")
cm_smote = confusion_matrix(y_test, y_pred_smote)
print(cm_smote)

# 可视化新的混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (After SMOTE)')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n图片 'confusion_matrix.png' 已成功保存到项目文件夹。")
plt.show()


import shap

# 步骤1: 创建一个SHAP解释器 (Explainer)
# 对于逻辑回归这类线性模型，使用 LinearExplainer 效率最高
explainer = shap.LinearExplainer(log_reg_model_smote, X_train_smote)

# 步骤2: 计算测试集样本的SHAP值
# 这一步会为X_test中的每一个样本，计算出每个特征对预测结果的贡献值
shap_values = explainer.shap_values(X_test)
print("\n--- [正在生成并保存SHAP模型解释图...] ---")

# 步骤1: 先绘制SHAP图，但设置show=False，暂时不显示它
shap.summary_plot(shap_values, X_test, show=False)

# 步骤2: [核心] 使用 plt.savefig() 保存我们刚刚绘制好的图
plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')

# 步骤3: 手动调用 plt.show() 将图表显示在屏幕上
plt.show()

print("图片 'shap_summary_plot.png' 已成功保存到项目文件夹。")
