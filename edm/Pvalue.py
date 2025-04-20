import numpy as np
from scipy.stats import ttest_ind
import pandas as pd

gen_data = pd.read_excel("generated_data_2avg_test.xlsx", header=None).values 

real_data = np.loadtxt("test_type123.txt", delimiter=",")

# 逐维度计算t检验的P值
p_values = []
for dim in range(7):  # 遍历7个维度
    t_stat, p_val = ttest_ind(real_data[:, dim], gen_data[:, dim])
    p_values.append(p_val)

# 格式化输出P值（保留3位小数）
print("P值（t检验） & " + " & ".join([f"{p:.3f}" for p in p_values]) + " \\\\")