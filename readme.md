# DMforICBpre: 利用扩散模型进行ICB疗效预测

## 项目简介
本项目通过扩散模型（Diffusion Models）实现免疫检查点阻断治疗（ICB）疗效的预测与特征组合探索，包含以下核心模块：
- **条件生成模型**（EDM文件夹）：基于条件生成的预测模型
- **条件概率建模**（ScoreMatching文件夹）：基于条件概率的预测模型
- **特征组合探索**（SM_traverse文件夹）：遍历寻找新的有效特征组合

## 环境依赖
- Python 3.10.14
- PyTorch 2.5.0
- 安装依赖：`pip install -r requirements.txt`

## 代码结构
```
├── EDM/ 
│ ├── train.py # 模型训练脚本
│ └── test.py # 生成结果测试
├── ScoreMatching/ 
│ ├── train.py # 训练
│ └── test.py # 概率计算测试
└── SM_traverse/
├── train.py # 针对所有特征组合训练模型
└── test.py # 对所有模型测试
```
