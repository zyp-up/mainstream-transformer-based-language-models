# 基于Transformer的三大主流语言模型架构

![思维导图概览](https://storage.googleapis.com/static.aiforpro.com/1689264930263-12826/orignal_033d45c8188ea6891398bb89280d0752.png)

## 概述

Transformer模型自2017年被提出以来，已成为自然语言处理（NLP）领域的基石。其独特的自注意力机制（Self-Attention）使其能够高效地处理长距离依赖关系。基于其核心架构，衍生出了三种主流的语言模型范式：**Encoder-only**、**Decoder-only** 和 **Encoder-Decoder**。

- **Encoder-only (代表: BERT)**：采用Transformer的编码器部分，因其核心机制（双向自注意力）和核心训练目标（掩码语言模型）都是为了一个目标而设计的：一次性地、全局性地捕捉并编码整段文本的深层上下文信息，故天然适合于理解型任务，如文本分类、实体识别、情感分析等。其通过上下文双向编码来理解文本。

- **Decoder-only (代表: GPT系列)**：采用Transformer的解码器部分，专注于生成式任务，如文本生成、对话系统、文章撰写等。其通过自回归（Auto-regressive）的方式，根据上文生成下一个词。

- **Encoder-Decoder (代表: T5, BART)**：采用完整的Transformer架构，包含编码器和解码器。核心思想就是将几乎所有NLP的下游任务都统一为一种“文本到文本”（Text-to-Text）的格式。常用于序列到序列（Seq2Seq）任务，如机器翻译、文本摘要等。

## Bert (Encoder-only)
[Bert论文原文](https://arxiv.org/abs/1810.04805)

![BERT模型整体结构](https://github.com/zyp-up/mainstream-transformer-based-language-models/raw/main/assets/bert1.png)

### 预训练
- **数据集构造**：
![数据集构造](https://github.com/zyp-up/mainstream-transformer-based-language-models/raw/main/assets/bert2.png)

   1. **输入序列构造**
![输入序列](https://github.com/zyp-up/mainstream-transformer-based-language-models/raw/main/assets/bert3.png)

   2. **Segment ID构造**
![Segment ID 构造](https://github.com/zyp-up/mainstream-transformer-based-language-models/raw/main/assets/bert4.png)

   3. **Positional Encoding**
定义一个可学习的参数矩阵：$E_{pos} \in \mathbb{R}^{seq\_lenth \times d_{hidden}}$
![Positional Encoding 细节](https://github.com/zyp-up/mainstream-transformer-based-language-models/raw/main/assets/bert5.png)

#### 两个预训练任务
1. **MLM (Masked Language Model)**

    - **Mask 构造**
    ![Mask 构造规则](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/bert6.png?raw=true)
    - **原始输入**：`[Batch_size, seq_lenth]`
    - **Word Embedding**
    ![Word Embedding](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/bert8.png?raw=true)
    - **送入Transformer Encoder**
    ![顶层输出](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/bert10.png?raw=true)
    - 取出训练过程中已经随机挑选了 15% 的mask token ，每条序列大约有 `|M|` 个 mask。于是我们从顶层输出中选出这些位置：
    ![选取 Masked Token 的输出](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/bert11.png?raw=true)
    - **通过全连接层和Softmax预测**
    ![MLM 预测过程](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/bert12.png?raw=true)
    - **MLM 损失函数**
    ![MLM 损失函数](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/bert13.png?raw=true)

2. **NSP (Next Sentence Prediction)**
    - **NSP 任务图示**
    ![NSP 任务图示](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/bert7.png?raw=true)
    - **取CLS向量进行分类**
    ![取CLS向量](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/bert9.png?raw=true)

### 微调 (Fine-tuning)
#### 文本分类
- **文本分类模型结构**
![文本分类模型结构](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/bert14.png?raw=true)
- 对每个样本（句子/句子对），输出属于 K 个类别中每一类的概率分布。
![文本分类损失函数](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/bert15.png?raw=true)

#### 问答任务 (Q&A)
- **问答任务模型结构**
![问答任务模型结构](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/bert16.png?raw=true)
- **问答任务输出与损失**
![问答任务输出与损失](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/bert17.png?raw=true)

#### 序列标注
- **序列标注模型结构**
![序列标注模型结构](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/bert18.png?raw=true)
- **序列标注输出与损失**
![序列标注输出与损失](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/bert19.png?raw=true)

