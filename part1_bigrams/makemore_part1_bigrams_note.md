## 1. Bigram 的核心思想

- 只关注 **相邻两个字符** 的关系（local structure）。
- 任务：给定当前字符，预测下一个字符的分布。

## 2. 数据构建方式

以名字 *Emma* 为例：

- 引入开头 `<S>`、结尾 `<E>` 两个特殊符号，用统一符号 `<.>` 代替，以简化统计表格结构。
- 转换序列：`<S> e m m a <E>`
- 生成所有相邻 pair：
  - `<S>-e`
  - `e-m`
  - `m-m`
  - `m-a`
  - `a-<E>`

用一个 **27×27 tensor** 记录所有 pair 的计数（26 字母 + `<.>`）。

## 3. 基于计数的 Bigram 模型

1. 统计所有 pair 的出现次数
2. 将计数除以总和 → 得到概率分布
3. 用 `torch.multinomial` 对概率分布采样，按概率生成下一个字符

### 优点

- 结构简单、可解释性强。

### 缺点

- 上下文长度固定为 1。
- 若扩展到 trigram、4-gram… 表格维度爆炸，几乎不可扩展。

## 4. Loss Function：负对数似然 NLL

- likelihood = 所有预测概率的乘积，数值易下溢 → 使用 log
- log 之后求和：加法比乘法更稳定
- negative：使 loss 越小代表模型越好（minimization）

## 5. Model Smoothing（加平滑）

避免概率为 0 → `log(0)= -inf`。

方法：对所有计数加上 fake counts（如 +1）。
 本质：增加分布的平滑性，使模型不极端。

------

# Neural Network 版本的 Bigram 模型

## 1. 核心结构

- 输入是 one-hot 向量
- 线性层 $W \in \mathbb{R}^{27 \times 27}$
- softmax 输出概率分布

## 2. 与计数法的等价性

- one-hot 输入只有一个位置是 1，因此 `W @ onehot` 等价于取 W 的某一行。
- 这与“查表”的 count-based bigram 本质一样。
- W 的每一行可视为某个字符后续字符的 **log-count**。
- 对 W 的每行 exponentiate 再 normalize → 与之前的计数表一致。

## 3. 神经网络方法的优势

- 参数可通过梯度下降训练，而不只是统计。
- 可扩展到更复杂的上下文（加入 embedding、加入多层结构）。

## 4. NN 中的 smoothing/regularization

- 与"加 fake counts"一致的想法：
  - L2 正则 → 惩罚 W 偏离 0
  - dropout / weight decay → 让模型更平滑
- 最终效果都是：让输出分布不要太尖锐。

------

# PyTorch 相关概念整理

## 1. `torch.sum`

**格式：** `torch.sum(input, dim=None, keepdim=False)`

- `dim=None` → 对整个张量求和
- `dim=i` → 压掉第 i 维
- `keepdim=True` → 保留被压缩的维度（长度变 1），便于后续广播

典型用法：

- 每列求和：`x.sum(0)`
- 每行求和：`x.sum(1)`
- 保持维度：`x.sum(1, keepdim=True)` → 常用于 softmax、归一化操作

------

## 2. Broadcasting（自动扩展机制）

**规则（从最后一维往前对齐）：**

1. 维度相同 → OK
2. 一个维度为 1 → 可以扩展
3. 维度不同且都不是 1 → 不能广播

本质：逻辑扩展，不实际复制内存。

例子：

- `(2,3)` 与 `(3,)` → `(3,)` 视为 `(1,3)` → OK
- `(2,3)` 与 `(2,1)` → 第 2 维 1 扩展为 3 → OK
- `(2,3)` 与 `(3,2)` → ❌ 无法对齐

keepdim 的作用：
 让求和后的维度更容易广播回原 shape。

------

## 3. `torch.Tensor` vs `torch.tensor`

### `torch.tensor`

- 工厂函数
- 推荐使用
- 自动推断 dtype
- 不会创建未初始化张量

### `torch.Tensor`

- 类（Class）
- 用 `torch.Tensor(shape)` 会创建**未初始化内存**（不安全）
- 老式用法

总结：**新建张量用 `torch.tensor()`。**

------

## 4. One-hot：`torch.nn.functional.one_hot`

- 输入必须是 **整数张量**
- 输出 dtype = `long`
- 输出 shape = `input.shape + (num_classes,)`
- 若要进入神经网络 → `.float()`

例：

```
F.one_hot(torch.tensor([0,2,1]), num_classes=3)
```

------

## 5. `@` 运算符（矩阵乘法）

等价于 `torch.matmul`。

- `*` → element-wise
- `@` → 矩阵/向量乘法

行为：

- 1D @ 1D → 点积
- 2D @ 2D → 矩阵乘
- 3D+ → 按最后两维做 batched matmul

