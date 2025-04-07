# 知识蒸馏
## 核心功能
常规训练：计算学生模型的交叉熵损失（Cross-Entropy Loss）。
​知识蒸馏：通过教师模型生成软标签（Soft Targets），结合KL散度损失优化学生模型。
​动态蒸馏策略：支持多种策略控制何时使用教师模型（如基于熵、随机性、错误预测等）。
## 1、基础训练流程
### 1.1数据准备
```python
inputs, labels = batch
inputs, labels = Variable(inputs), Variable(labels)  # 转换为PyTorch变量
outputs = model(inputs)  # 学生模型前向传播
```
### 1.2常规损失计算
```python
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)  # 交叉熵损失
```
### 1.3知识蒸馏实现
|   策略类型     |  实现逻辑           |
|----------------|--------------------|
|**always**      |对所有样本使用教师模型|

```
print("hello")
```

