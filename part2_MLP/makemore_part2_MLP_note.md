- - # Makemore Part 2 — MLP 学习笔记
  
    ## 1. `torch.view()`：改变维度的最简单方式
    - PyTorch 内部把 tensor 存成一维数组，维度只是展示方式。
    - 改变维度如 `(32,3,2) -> (32,6)` 最直接的方法就是 `x.view(32, 6)`。
    - 本质：不复制数据，只改变“怎么看”。
  
    ---
  
    ## 2. 为什么使用 `F.cross_entropy()`
    **(1) 更高效**
    - 内部实现了 log-softmax + NLLLoss，不需要手动再写所有操作。
    - 类似 micrograd 中封装 `tanh`，能跳过一堆基础运算。
  
    **(2) 数值稳定性更好**
    - softmax 涉及 exp 操作，大正数容易溢出成 `inf`。
    - PyTorch 的处理方式：先减掉最大值 → 全部变成非正数 → 避免溢出。
  
    ---
  
    ## 3. Loss 不可能为 0 的原因
    - 对于预测第一个字母，不同名字的 label 都不一样，模型无法“完美确定”答案。
    - 语言模型在open-set任务中本来就无法达到绝对0 loss。
      - open-set任务：模型的输入可能对应很多合法输出、且无法穷举所有正确答案的任务。比如预测下一个字符、生成文本、对开放世界图片分类等。
    
  
    ---
    
    ## 4. Mini-batch 的作用
    - 每次只取 batch_size=32 的样本进行训练，提高速度、降低内存占用。
    - 比 full-batch 更能提供噪声，帮助跳出坏局部最优（local optima）。
  
    ---
    
    ## 5. 如何找到合适的 Learning Rate
    1. 先人为试几个，找一个可能范围。
    2. 在这个范围内用 *指数采样*（不是线性）生成几百个 lr 值。
    3. 对每个 lr 训练少量 step，记录 loss。
    4. 画出 “learning rate vs loss” 曲线，找到最佳 lr。
    5. 完整训练过程中通常会使用 **learning rate decay** 做小步微调收敛。
  
    ---
    
    ## 6. Train / Dev / Test Split
    - **Train（80%）**：训练模型参数。
    - **Dev / Validation（10%）**：调超参数（例如 lr、hidden size 等）。
    - **Test（10%）**：最终评估性能，只用一次。
  
    ---
    
    ## 7. 常见 Hyperparameters
    - Hidden layer neurons：100 → 300
    - Batch size：太小噪声大，太大收敛慢
    - Learning rate / lr decay
    - Embedding size：2 → 10
    - Context size（窗口长度）
    - 训练轮数（epochs）
    
    