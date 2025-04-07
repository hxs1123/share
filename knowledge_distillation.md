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
ce_loss = criterion(outputs, labels)  # 交叉熵损失
```
### 1.3知识蒸馏实现
**(1)动态蒸馏策略选择**
|   策略类型            |  实现逻辑                          |  适用场景         |
|---------------------- |---------------------------------- |------------------|
|**always**             |对所有样本使用教师模型               |  默认策略         |
|​**cutoff_entropy**     |仅对高熵样本（预测不确定）使用教师模型 | 困难样本重点学习   |
|​**random_entropy**     |按熵值概率随机选择样本               |平衡蒸馏与常规训练   |
|​**incorrect_labels**   |仅对错误预测样本使用教师模型          | 纠错学习          |

```python
if ask_teacher_strategy[0].lower() == 'always':
    mask_distillation_loss = torch.ByteTensor([True]*outputs.size(0))
elif ask_teacher_strategy[0].lower() == 'cutoff_entropy':
    entropy = [mhf.get_entropy(prob_out[idx_b, :]) for idx_b in range(prob_out.size(0))]
    mask_distillation_loss = torch.ByteTensor([entr > cutoff_entropy_value_distillation for entr in entropy])
```
**(2)损失计算**
​KL散度损失​（教师 vs 学生）
```python
outputsTeacher = teacher_model(volatile_inputs).detach()  # 冻结教师梯度
kl_loss = KLDivLossFunction(
    logSoftmaxFunction(outputs / temperature_distillation),
    softmaxFunction(outputsTeacher / temperature_distillation)
) * (temperature_distillation ​** 2)  # 温度缩放
```

加权总损失
```python
loss_masked = weight_teacher_loss * kl_loss + (1-weight_teacher_loss) * ce_loss
```





