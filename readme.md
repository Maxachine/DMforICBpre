# DMforICBpre: 利用扩散模型进行ICB疗效预测

## 项目简介
本项目通过扩散模型（Diffusion Models）实现免疫检查点阻断治疗（ICB）疗效的预测与特征组合探索，包含以下核心模块：
- **条件生成模型**（EDM文件夹）：基于扩散模型的条件数据生成
- **条件概率建模**（ScoreMatching文件夹）：基于分数匹配的条件概率估计
- **特征组合探索**（SM_traverse文件夹）：遍历潜在空间发现有效特征组合

## 环境依赖
- Python 3.10.14
- PyTorch 2.5.0
- 安装依赖：`pip install -r requirements.txt`

## 代码结构
├── EDM/ # 条件生成模型
│ ├── train.py # 模型训练脚本
│ └── test.py # 生成结果测试
├── ScoreMatching/ # 条件概率模型
│ ├── train.py # 分数匹配训练
│ └── test.py # 概率估计验证
└── SM_traverse/ # 特征组合探索
├── train.py # 训练潜在空间映射
└── test.py # 执行特征遍历搜索
