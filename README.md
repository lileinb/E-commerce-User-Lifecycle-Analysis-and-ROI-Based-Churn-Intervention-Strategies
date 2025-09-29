# 电商用户生命周期分析与流失预警模型
### E-commerce User Lifecycle Analysis and ROI-Based Churn Intervention Strategies

---

### 1. 项目概述 (Project Overview)

本项目是一个端到端的数据分析与机器学习项目，旨在模拟真实商业场景，解决电商平台在用户增长放缓背景下的核心痛点：**如何通过数据驱动实现精细化用户运营，识别高价值用户，并提前预测和干预用户流失，最终提升整体用户生命周期价值(LTV)。**

项目从超过54万行的原始交易流水数据出发，完成了从**数据清洗、探索性分析(EDA)、RFM用户价值分层、交互式可视化看板搭建、LTV估算、流失用户预测建模，到最终模型解释与归因分析**的全链路流程。


### 2. 商业问题与分析目标 (Business Problem & Objectives)

* **核心痛点:** 获客成本日益增高，如何有效激活、留存现有用户，并最大化其商业价值，成为业务增长的关键。
* **分析目标:**
    1.  **用户分层:** 如何对现有用户进行科学、有效的价值分层，以实现差异化运营？
    2.  **价值量化:** 如何评估不同用户群体的长期价值(LTV)，以便合理分配运营资源？
    3.  **流失预测:** 能否在用户彻底流失前，提前识别出高风险用户？
    4.  **归因分析:** 驱动用户流失的关键行为是什么？如何基于此制定有效的干预策略？

### 3. 技术栈 (Tech Stack)

* **数据处理与建模:**
    * `Python 3.x`
    * `Pandas`: 数据清洗、转换与处理
    * `NumPy`: 高性能数值计算
    * `Scikit-learn`: 机器学习（逻辑回归、数据划分）
    * `Imbalanced-learn`: 处理数据不平衡问题 (SMOTE)
* **可视化与模型解释:**
    * `Matplotlib` & `Seaborn`: 探索性数据分析与静态图表绘制
    * `Tableau Public`: 交互式仪表板搭建与商业洞察呈现
    * `SHAP`: 机器学习模型可解释性分析与归因

### 4. 项目流程与文件结构 (Project Workflow & File Structure)

本项目严格遵循专业的数据分析工作流，代码结构清晰，职责分离：

1.  **`Data_Cleaning.py`**:
    * 负责项目的**第一阶段：数据准备与特征工程**。
    * **输入:** `online_retail.csv` (原始数据)
    * **流程:**
        * 探索性数据分析(EDA)，识别数据问题。
        * 数据清洗（处理缺失值、退货订单、无效价格）。
        * 构建RFM模型（计算R,F,M绝对值 -> 打分 -> 分层命名）。
        * 估算用户历史利润贡献，为LTV分析做准备。
        * 构建流失预测所需的高级特征（平均客单价、购买周期稳定性、品类多样性）。
        * 定义流失标签 (Churn)。
    * **输出:** `rfm_final_for_modeling.csv` (干净、完整的建模数据) 和 `rfm_for_tableau.csv` (可视化专用数据)。

2.  **`Feature_Project.py`**:
    * 负责项目的**第二阶段：机器学习建模与解释**。
    * **输入:** `rfm_final_for_modeling.csv`
    * **流程:**
        * 加载已处理好的数据。
        * 划分训练集与测试集。
        * 使用SMOTE技术处理训练集的数据不平衡问题。
        * 训练逻辑回归分类模型。
        * 在测试集上评估模型性能（生成分类报告和混淆矩阵）。
        * 使用SHAP框架对模型进行解释，探究流失成因。
    * **输出:** `confusion_matrix.png` 和 `shap_summary_plot.png` (模型性能与解释的可视化结果)。

3.  **Tableau工作簿 (`.twbx`)**:
    * 负责项目的**第三阶段：商业洞察可视化**。
    * **输入:** `rfm_for_tableau.csv`
    * **流程:**
        * 创建“用户数量分布”、“用户价值贡献”、“RFM行为矩阵”三个核心视图。
        * 整合为交互式的`RFM用户价值分析看板`。

### 5. 核心发现与业务洞察 (Key Findings & Business Insights)

#### 5.1 用户价值分层洞察 (From Tableau)
* **价值高度集中:** “高价值新客户”与“重要价值客户”这两个群体虽然数量占比不高，但贡献了绝大部分的销售收入，完美印证了“二八定律”。
* **用户结构:** 平台拥有大量“低价值客户”，同时也有相当规模的“潜力客户”和“新客户”，为后续的精细化运营提供了充足的目标人群。

#### 5.2 用户流失归因洞察 (From SHAP)
通过SHAP模型解释，我们识别出驱动用户流失的三大关键行为信号：
1.  **忠诚度与价值感下降 (`F_Score`, `M_Score`):** 用户的购买频率和消费等级是其留存意愿最强的决定因素。
2.  **购买行为不稳定 (`PurchaseIntervalStd`):** 用户的购买周期变得毫无规律，是其即将流失的最危险信号。
3.  **兴趣广度收窄 (`UniqueProductsCount`):** 用户购买的商品品类越单一，其流失风险越高。

### 6. 如何运行 (How to Run)

1.  **环境准备:**
    ```bash
    pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib shap
    ```
2.  **数据清洗与特征工程:**
    * 将原始数据 `online_retail.csv` 放置在项目根目录。
    * 运行 `Data_Cleaning.py` 脚本：
        ```bash
        python Data_Cleaning.py
        ```
    * 脚本将自动生成 `rfm_final_for_modeling.csv` 和 `rfm_for_tableau.csv`。

3.  **模型训练与解释:**
    * 运行 `Feature_Project.py` 脚本：
        ```bash
        python Feature_Project.py
        ```
    * 脚本将自动输出模型评估报告，并生成 `confusion_matrix.png` 和 `shap_summary_plot.png`。

4.  **交互式可视化:**
    * 安装 Tableau Public 或 Tableau Desktop。
    * 打开软件，连接到数据源 `rfm_for_tableau.csv`，即可开始复现或探索可视化看板。

### 7. 总结 (Conclusion)

本项目通过一个完整的端到端分析流程，不仅成功构建了一个能够有效识别潜在流失用户的机器学习模型（召回率达72%），更重要的是，通过模型解释深刻洞察了导致用户流失的关键行为因素。这些基于数据的洞察，能够直接指导业务部门制定出更精准、ROI更高的用户挽留策略，从而实现数据价值的商业闭环。
