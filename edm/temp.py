import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics
import pdb

# 读取预测 response
file_path = 'generated_data_2avg_test.xlsx' 
df = pd.read_excel(file_path,header=None)
pred = df.iloc[:, 6].tolist()

# 载入真实 response
file_path = 'test_type123.txt'  
response = []

with open(file_path, 'r') as f:
    for line in f:
        # 假设每行数据是空格或逗号分隔的
        data = line.strip().split(',')  # 如果是逗号分隔，使用 .split(',')
        response.append(float(data[6]))  # 第七维数据在索引6位置

#-------------------------------------------------------------
auc = roc_auc_score(response,pred)
print('auc:',auc)

#--------------------------------------------------------------
thresholds = np.linspace(0, 1, 5000)

# 用于存储不同阈值下的 F1 score
f1_scores = []

# 遍历每个阈值
for threshold in thresholds:
    pred01 = [1 if p > threshold else 0 for p in pred]
    f1 = f1_score(response, pred01)
    f1_scores.append(f1)

# 找到最大 F1 和对应的 threshold
max_f1 = max(f1_scores)
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"Maximum F1 score: {max_f1}")
print(f"Best threshold: {best_threshold}")
#------------------------------------------------------------

# 混淆矩阵计算
# pred01 = [1 if p > 0.35127025405081014 else 0 for p in pred]
# cm = metrics.confusion_matrix(response, pred01)
# TN, FP, FN, TP = cm.ravel()

# print(TN)
# print(FP)
# print(FN)
# print(TP)
# 362
# 26
# 90
# 37