# store_sales
Prodict the store sales states of the future based on mathine learning.

### 1. 数据准备

将train与transactions数据集放到/dataset/train下

将test数据集放到/dataset/test下

其余数据集放到/dataset/extra下

### 2. 预处理

/pretreatment下包括数据预处理的全部脚本

**包括**：
- dataload: 数据加载
- datamachine: 数据处理
- datavisualize: 数据可视化[数据分析]

### 3. 运行 TRAIN

运行run.py开始进行数据预处理和训练

### 4.模型保存

- /save/loss: 记录模型运行时的loss值

- /save/model: 记录epoch中最佳模型

- /save/predictions: 记录模型预测折线图

**部分设置可在config.yml中进行调整**

官方数据集：https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data