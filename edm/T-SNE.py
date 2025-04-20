import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.font_manager import FontProperties


font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', size=12) 

X_real = pd.read_excel("generated_data_2avg.xlsx", header=None).values 

X_fake = np.loadtxt("train_type123.txt", delimiter=",")

X_combined = np.vstack([X_real, X_fake])
X_2d = TSNE(n_components=2, random_state=42).fit_transform(X_combined)  # 固定随机种子保证可复现

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(X_2d[:len(X_real), 0], X_2d[:len(X_real), 1], c='blue', alpha=0.5, label="生成数据")
plt.scatter(X_2d[len(X_real):, 0], X_2d[len(X_real):, 1], c='red', alpha=0.5, label="Chowell_train数据")
# plt.title("生成数据与测试数据的t-SNE可视化", fontproperties=font)
plt.xlabel("t-SNE 维度 1", fontproperties=font)
plt.ylabel("t-SNE 维度 2", fontproperties=font)
plt.legend(prop=font)
plt.grid(alpha=0.3)

plt.savefig("tsne_plot.pdf", bbox_inches='tight', dpi=300,  pad_inches=0.1)
plt.show()