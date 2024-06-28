## MEMO模型解析

**赖继杰 21311193**	Contact: laijj23@mail2.sysu.edu.cn

### MEMO简介

MEMO（Memory-efficient Expandable MOdel，内存高效的可扩展模型）是一种由Da-Wei Zhou等人提出的基于模型的类增量学习方法，该模型根据新数据对已有的网络进行扩展和剪枝操作，以使模型动态地匹配增量学习任务的需求。

原论文：[A Model or 603 Exemplars: Towards Memory-Efficient Class-Incremental Learning](https://arxiv.org/abs/2205.13218)

原Github网址：https://github.com/wangkiw/ICLR23-MEMO

### 项目结构介绍

项目主体结构和代码如下：

```plaintext
lab_ML_MEMO/
├─ convs/					# 主干网络实现
│  ├─ cifar_resnet.py
│  ├─ conv_cifar.py
│  ├─ conv_imagenet.py
│  ├─ linears.py			# 线性全连接层的实现
│  ├─ memo_cifar_resnet.py	# 数据集CIFAR100适用的ResNet主干网络，包括通用块和专用块的构建
│  ├─ memo_resnet.py
│  ├─ model2examplar.py
│  ├─ resnet.py
│  ├─ resnet10.py
│  ├─ ucir_cifar_resnet.py
│  ├─ ucir_resnet.py
│  └─ __init__.py
├─ exps/
│  └─ memo.json				# MEMO模型初始参数
├─ models/
│  ├─ base.py				# 基础模型，包含计算和构建样例部分
│  └─ memo.py				# MEMO主体部分，包括初始训练阶段和增量训练阶段
├─ reprod_results/			# 复现实验结果
│  ├─ conf_matrix/			# 混淆矩阵结果
│  ├─ fair/					# cnn-topk和nme-topk指标
│  ├─ display.py			# 报告折线图生成代码
│  ├─ result1.log			# 报告实验1运行日志
│  ├─ result2.log			# 报告实验2运行日志
│  └─ result3.log			# 报告实验3运行日志
├─ utils/
│  ├─ data.py				# 下载和设置数据集
│  ├─ data_manager.py		# 数据集处理
│  ├─ factory.py			# 调用对应模型（MEMO）
│  ├─ inc_net.py			# 增量网络的具体实现，包含动态网络
│  ├─ model2exemplar.py
│  └─ toolkit.py
├─ compute_exemplar.py
├─ main_memo.py				# 项目入口，解析命令行
├─ report.pdf				# 实验报告
└─ trainer.py				# 设置参数，存储日志，调度学习任务
```

### 复现实验

Base-0, Inc-10的基础设置：
```python
python main_memo.py -model memo -init 10 -incre 10 -p fair --train_base -d 0 1 2 3
```

在此基础上添加和修改参数，得到以下复现实验：

**复现实验1：**(添加`--init_epoch 71 --epochs 51 -ms 3312 -net memo_resnet32`)

```python
python main_memo.py -model memo -init 10 -incre 10 -p fair --train_base -d 0 1 2 3 --init_epoch 71 --epochs 51 -ms 3312 -net memo_resnet32
```

**复现实验2：**(在上面的基础上修改`-ms 2495`)

```python
python main_memo.py -model memo -init 10 -incre 10 -p fair --train_base -d 0 1 2 3 --init_epoch 71 --epochs 51 -ms 2495 -net memo_resnet32
```

**复现实验3：**（在上面的基础上修改`-net memo_resnet14_cifar`）

```python
python main_memo.py -model memo -init 10 -incre 10 -p fair --train_base -d 0 1 2 3 --init_epoch 71 --epochs 51 -ms 2495 -net memo_resnet14_cifar
```

### 总结

在实验报告中，我们并没有过多地直接照搬原论文作者的理解，而是根据自己的理解重新定义了类增量学习和损失函数。并且根据自己的理解制作了一些MEMO的结构和运作示意图。

我们对原作者的代码进行了小幅度的修改（比如torch版本兼容，变量类型调整等），以使得代码能够在我们的电脑环境上正常运行。受限于精力和算力限制，我们并没有进行太多的优化以及复现实验，这是一大遗憾。

---

吐槽：
要是我有足够的算力这整个作业我可以从头到尾一个人完成😢，但是我的搭档几乎真的除了用他的电脑跑实验结果以外什么事情都没干啊😤，甚至实验结果的折现图都是我写代码画出来的😫（其实可以把上面的所有代词“我们”改成“我”），请老师/助教明鉴😭😭😭！
