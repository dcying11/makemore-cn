## fixing the initial loss

### 1. 问题
- 随机初始化可能让权重的 scale 偏大（confidently wrong）。  
- 结果是最初 logits 过大 → 初始 loss 异常高。  
- 训练一开始模型并没有在学习结构，只是在处理数值过大的后果。

### 2. 分析
- 前几步优化器做的事情主要是 **squash logits**：把整体幅度压小。  
  - 这是 easy gain，因为只需整体缩放，不需理解数据模式。
- 真正的 hard gain 是后续：  
  - 调整 logits 的 **相对大小**  
  - 即学习哪些类别应该更高、哪些更低（真正的 pattern learning）

### 3. 解决
- 缩小初始化 scale，让训练不用浪费前几步去“压缩数值”：  
  - `w2 *= 0.01`  
  - `b2 = 0`  
- 效果：初始 logits 合理，模型更快进入“学习结构”的阶段。

--------

## fixing the saturated tanh

### 1. 问题
- forward pass 中大量 tanh 输出为 -1 或 1，说明神经元处在 tanh 的 flat tail。
- tanh 的 local gradient = 1 - t^2  
  - 当 t ≈ ±1 时，梯度≈0 → 反向传播时梯度直接消失。
- 结果：许多神经元在训练早期就完全不更新。

### 2. dead neuron 的两种情况

#### （1）初始化导致的 dead neuron
- tanh 的有效区间在中间（输出≈0 时梯度最大）。
- 若初始化权重/偏置过大，使输入始终落在饱和区：
  - tanh(100) ≈ 1（grad≈0）
  - tanh(-100) ≈ -1（grad≈0）
- 神经元输出永远接近 ±1，梯度长期为 0 → **dead neuron**。

#### （2）优化过程导致的 dead neuron（learning rate 太大）
- 学习率过大 → 某一步更新过猛 → 参数跳到极端值。
- 神经元被“打飞”到饱和区，不再回到可学习区域 → **永久死掉**。

### 3. 类似会出现问题的激活函数
- **sigmoid / tanh**：两端饱和，出现 flat tail → 梯度消失。  
- **ReLU**：负区梯度=0，若输入始终<0 → dead ReLU。

### 4. 如何解决
- 减小初始化 scale：`w1 *= 0.2`, `b1 = 0.01`  
  → 让神经元更可能落入 tanh 的有效区间。
- 结果：dead neuron 显著减少，训练更有效，loss 更容易下降。

--------

## calculating the init scale: “Kaiming init”

之前我们手动乘以 0.2 或 0.01 来减小初始化的 scale，但实际上没有人会手动这么做。

### 1. 问题

- 线性层会把输入分布放大或缩小，若不控制初始化，前向激活可能 **层层膨胀** 或 **层层衰减**。
- 结果：logits 尺度失控，梯度爆炸 / 梯度消失，神经元落入激活函数的无效区（如 tanh saturation）

### 2. 分析

#### （1）方差在前向传播中的变化

- 输入 $x\sim \mathcal N(0,1)$，权重 $W\sim\mathcal N(0,1)$。
- 线性变换 $y = xW$ 的输出 std 会变大，例如从 1 → 3。
- 若把 $W$ 放大，std 会继续扩大；若缩小，std 会缩小。
- **目标：让每层输出的标准差保持在 ~1，不爆不灭。**

#### （2）数学结果：fan-in 决定初始化尺度

- 推导可得：
  $$
  \mathrm{std}(W)=\frac{1}{\sqrt{\text{fan-in}}}
  $$

- fan-in = 每个神经元看到的输入维度（例如 10、30 等）。

### 3. 加上激活函数的影响（gain）

激活函数会改变分布，需要额外补偿因子 gain。

- 目的：防止方差在层间不断变大（爆炸）或变小（消失），保持可学习性。

- gain 太大：初始化过大 → 激活饱和 → 梯度≈0 → dead neuron、训练变慢。

- gain 太小：初始化过小 → 激活接近 0 → 梯度弱 → 表达能力不足、欠拟合。

- 直观理解：tanh 本身会把数压向 0（压缩分布），需要 gain 把分布重新拉开。但 gain 不能过大，否则又把神经元推入 tanh 的饱和区。经验上 tanh 的增益 5/3 可以很好地平衡这两种力量。

- 每种非线性都有经验最优 gain

  - **ReLU**：丢掉一半分布 → gain = √2

  - **tanh**：挤压尾部 → gain = 5/3

  - **linear/identity**：gain = 1

最终标准差应为：
$$
\text{std}(W)=\frac{\text{gain}}{\sqrt{\text{fan-in}}}
$$

### 4. 解决：标准做法是Kaiming initialization

PyTorch 的 `torch.nn.init.kaiming_normal_` 已内置：

- `mode="fan_in"`（默认）：保持前向激活稳定
- `nonlinearity="relu"`, `"tanh"` 等：自动计算 gain

大多数情况下，只需用默认 fan_in + 对应 nonlinearity 即可。

### 5. 实践

例如 tanh 且 fan-in=30：

```
gain = 5/3
std = gain / (30 ** 0.5)
W = torch.randn(... ) * std
```

比手动 `*0.2` 或 `*0.01` 更数学化、更稳定、更通用。

### 其实现代网络对初始化不那么敏感

因为出现了稳健训练技巧，使得即便初始化不完美，训练仍能稳定。

- Residual connections（极大缓解梯度消失/爆炸）
- BatchNorm / LayerNorm / GroupNorm（动态标准化激活）
- 更好的 optimizer（Adam、RMSprop）

- "Kaiming init" paper: [https://arxiv.org/abs/1502.01852](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbU5zOUtEbGZXaUpIOC1WeWZNWThOS1g4WEJ6d3xBQ3Jtc0tsSXl3Z0hCOTZLRDFZQXFMeWE0TnhPLTJoU3hWYmZ4bXkyVUpMTFB4NDllR1RiZWxhUHJsUlpsczk1S1FWVGhEZXhvUkRZQzRWcWR2eWxjTGwzOVRFRHpwckVlc3k4cUk5V0pmaE9oc0RVM2RFSUhNSQ&q=https%3A%2F%2Farxiv.org%2Fabs%2F1502.01852&v=P6sfmUTpUmc)

-----

## Batch Normalization

Batch Norm 用于稳定深层网络的训练过程，通过对每个 batch 的激活分布做归一化，使训练更快、更稳定。

### 1. 核心机制

**(1) Normalize：归一化 pre-activation**

- 对线性层输出（pre-activation）按 batch 计算均值与方差。
- 将其处理为均值 0、方差 1 的分布。

**(2) Scale & Shift：恢复表达能力**

- 初始化时希望 pre-activation 接近 unit Gaussian，但训练中保持固定分布会限制模型表达能力。
- 因此在归一化后加入可学习参数 γ（scale）和 β（shift），初始分别为 1、0。

**(3) 使用位置**

- 一般放在线性层 / 卷积层等 **有乘法操作(product)的层之后**。

### 2. BN 的 regularizing effect（正则化效果）

**BN 会让样本在同一个 batch 内产生依赖（coupling）**：

- BN 在计算均值/方差时使用整个 batch 的数据。
- 因此每个样本的归一化结果依赖于当前 batch 的样本组成/它们的统计量
- 这会带来 **抖动（jitter）** —— forward/backward 的值不再固定，而是对 batch 抽样敏感。
- 这种 noise 会起到类似正则化的作用，有助于防止过拟合。

**缺点：** 由于 coupling，会出现一些 bug，也催生了 LayerNorm、GroupNorm 等更稳定的替代方案。

### 3. 训练阶段 vs 推理阶段

- **训练阶段：** 每个 batch 都计算自己的 mean/std。  

- **推理阶段：** 使用 running mean/std，而非当前 batch 的统计量。

- 更新方式（动量 momentum）：

  - bnmean_running = 0.99 * bnmean_running + 0.01 * bnmean_i
  - bnstd_running  = 0.99 * bnstd_running  + 0.01 * bnstdi

  - `0.01` 为 momentum。

  - 好处：
    - 训练时自动更新统计量，无需额外计算。
    - 推理时可对单个样本进行归一化，而不是依赖 batch。

### 4. 与 Linear 层的关系

因为 BN 会在归一化过程中消掉偏置项，因此：

- 若使用 BN，通常将线性层设为：`bias=False`  ，真正起作用的偏置是 BN 的 β（shift）。

### 5. PyTorch 相关补充

- [BatchNorm1d 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)

  - `eps`：防止分母为 0，增强数值稳定性。

- [Linear 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)

  - `torch.nn.Linear` 参数：若配合 BN：`bias=False`

  - Linear 的 weight 初始化：按 `1 / (fan-in)^2` 的范围从 uniform 采样。


-------

## update-to-data ratio

### 1. 为什么 gradient-to-data ratio 不够有用？

因为真正决定训练是否稳定的不是梯度的大小，而是 **参数实际会被更新多少**：
$$
\text{update} = \text{learning rate} \times \text{gradient}
$$
所以关键是：
$$
\text{update-to-data ratio} = 
\frac{\text{参数更新量}}{\text{参数本身的大小}}
$$
这个比值才决定训练是不是太激进或太慢。

### 2. 什么是 update-to-data ratio？

对每个参数张量求：

- 更新量的标准差（lr × grad）
- 参数本身的标准差（data）
- 取两者的比值
- 再取 log10（方便可视化）

含义：**更新量占参数本身大小的多少？**

### 3. 为什么要关注它？

- 如果更新量 **太大**（ratio ≫ 10⁻³），参数会被过度修改 → 不稳定、震荡
- 如果更新量 **太小**（ratio ≪ 10⁻³），参数几乎不动 → 学不动、训练太慢

经验上：**理想的 update-to-data ratio ≈ 10⁻³（log10 = –3）**

也就是：**每次更新不要超过参数本身大小的千分之一。**

### 4. 为什么最后一层一开始 ratio 特别大？

因为在初始化时，我们把最后一层权重 **乘以 0.1**（让 softmax 不要太自信 confidently wrong）。
这会导致：参数本身数值变得很小，但梯度正常大小，所以 ratio = update / data 会临时变得特别大。

后来训练几步之后，参数逐渐变大，ratio 就稳定下来。

### 5. 如何使用这个 ratio？

在训练时观察所有参数的 update-to-data ratio：

- **高于 10⁻³太多** → 学习率太大，更新太激进
- **低于 10⁻³太多** → 学习率太小，参数不动

所以它是用来判断学习率是否合适的一个关键指标。