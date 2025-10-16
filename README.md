# 基于Transformer的三大主流语言模型架构

## Update
**v1.0 (2025-10-16):**
- 全面梳理了基于Transformer的三大主流语言模型架构（Encoder-only, Decoder-only, Encoder-Decoder）
- 详细拆解了 **BERT** 的预训练任务（MLM, NSP）与微调流程。
- 阐述了 **T5** 将所有NLP任务统一为Text-to-Text范式的核心思想。
- 深入探讨了 **GPT系列** 的演进，特别是 **InstructGPT** 中基于人类反馈的强化学习（RLHF）的全流程细节。

![思维导图概览](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/%E5%9F%BA%E4%BA%8Etransformer%E7%9A%84%E4%B8%89%E5%A4%A7%E4%B8%BB%E6%B5%81%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84.png?raw=true)

## 概述

Transformer模型自2017年被提出以来，已成为自然语言处理（NLP）领域的基石。其独特的自注意力机制（Self-Attention）使其能够高效地处理长距离依赖关系。基于其核心架构，衍生出了三种主流的语言模型范式：**Encoder-only**、**Decoder-only** 和 **Encoder-Decoder**。

- **Encoder-only (代表: BERT)**：采用Transformer的编码器部分，因其核心机制（双向自注意力）和核心训练目标（掩码语言模型）都是为了一个目标而设计的：一次性地、全局性地捕捉并编码整段文本的深层上下文信息，故天然适合于理解型任务，如文本分类、实体识别、情感分析等。其通过上下文双向编码来理解文本。

- **Decoder-only (代表: GPT系列)**：采用Transformer的解码器部分，专注于生成式任务，如文本生成、对话系统、文章撰写等。其通过自回归（Auto-regressive）的方式，根据上文生成下一个词。

- **Encoder-Decoder (代表: T5, BART)**：采用完整的Transformer架构，包含编码器和解码器。核心思想就是将几乎所有NLP的下游任务都统一为一种“文本到文本”（Text-to-Text）的格式。常用于序列到序列（Seq2Seq）任务，如机器翻译、文本摘要等。

---

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

---

## T5 (Encoder-Decoder)
[T5论文原文](https://arxiv.org/abs/1910.10683)
将所有NLP任务都统一为一种“文本到文本（`text-to-text`）”的格式。无论是翻译、分类、回归还是摘要，输入都是文本，输出也都是文本。

1.  **数据构造**
    1.  **初始数据构造**
        这是T5框架的第一步，也是最巧妙的一步。所有原始数据都需要被预处理成带有任务前缀的输入字符串和目标输出字符串。

| 任务类型 | 原始数据 | 构造后的输入 (Input String) | 构造后的标签 (Target String) |
| :--- | :--- | :--- | :--- |
| **机器翻译** | (源语言句子, 目标语言句子) | `translate English to German: ` + 源语言句子 | 目标语言句子 |
| **文本分类** | (句子, 标签) | `cola sentence: ` + 句子 | 标签对应的文本 (e.g., `acceptable`) |
| **摘要** | (长篇文章, 摘要) | `summarize: ` + 长篇文章 | 摘要 |
| **语义相似度** (回归) | (句子1, 句子2, 分数) | `stsb sentence1: ` + 句子1 + ` sentence2: ` + 句子2 | 分数对应的字符串 (e.g., `3.8`) |

2.  **预训练（跨度破坏 Span Corruption）**
    T5与BERT的预训练方式不同。例如，以 `Thank you <X> me to your party <Y> week.` 为输入，模型需要用自回归的方式逐个生成预测 `<X> for inviting <Y> last <Z>`。这相当于在解码器中输入一个起始符（start token），然后让它预测出句子中所有被破坏的片段，达到“完形填空”的效果。

---

## GPT系列(Decoder-only)

### GPT-1
[GPT-1论文原文](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

#### 架构细节：
- **层数 (Layers):** 12 层 Transformer Decoder 模块。
- **隐藏层维度 (Dimensionality):** 768 维。
- **注意力头数 (Attention Heads):** 12 个。
- **前馈神经网络维度 (Feed-Forward Networks):** 内部前馈神经网络的维度为 3072。

#### 无监督预训练 (Unsupervised Pre-training)

- **任务本质：** 从一个连续的、无边界的文本流中学习语言的内在模式。它的任务不是“完成一个句子”，而是“在任何给定的上下文中，预测下一个词是什么”。
与transformer训练不同，没有`eos`、`bos`、`pad`等特殊token（pad也可以有，只需要在后续mask掉，取决于训练需求）。

- **为什么不需要 `bos` 和 `eos`？**
    - **数据是连续的流:** 想象一下，模型正在阅读一本几百万字的小说。我们从中随机截取一段512个词的片段来训练它。
    - **没有逻辑上的“开始” (`bos`):** 这个片段的第一个词，很可能是上一句话的中间部分。在这里强行加上`bos`是人为且错误的，因为它并不是一个真正意义上的句子开头。模型需要学习的是“无缝衔接”，而不是“从头开始”。
    - **没有逻辑上的“结束” (`eos`):** 这个片段的最后一个词后面，小说还在继续。模型的任务是基于这512个词，去预测第513个词。它不需要一个停止信号，因为它的工作永远是“继续下去”。训练的目标也不是让它生成一个完整的单元然后停止。
    - 从自然连续的文本数据流截取文本自然也就没有pad。

- **模型输入 (Input):**
    - **形状:** `[B, T_dec]`，即 `[64, 512]` 的整数张量（Tensor）。

- **模型输出 (Output):**
    - **形状:** `[B, T_dec, Vocab_Size]`，即 `[64, 512, 40000]` 的浮点数张量。
    - **具体样子:** 对于输入的每一个 token 位置，模型都会预测词汇表中所有 40,000 个 token 作为下一个词的概率得分（logits）。

- **标签 (Label):**
    - **核心思想:** 标签就是输入序列本身向左平移一位。这被称为“自监督学习”（Self-supervised Learning）。
    - **形状:** `[B, T_dec]`，即 `[64, 512]` 的整数张量。

#### 有监督微调 (Supervised Fine-tuning)
![GPT-1 微调示意图](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/gpt1-1.png?raw=true)

- **整体流程**
    1. 在预训练好的 Transformer 模型顶部添加一个全新的、任务专属的线性层 (Linear Layer)。
    2. 根据任务类型，将输入数据（如句子、句子对）构造成一个单一的、带特殊指令符的 token 序列。
    3. 将该序列输入 Transformer 模型，进行前向传播。
    4. 只取出序列最后一个指令符 (`Extract`) 所对应的最终输出向量（Hidden State）。
    5. 将这个向量作为整个序列的特征表示，送入顶部的线性层，得到分类结果。
    6. 使用任务的标签计算损失，并反向传播，更新整个模型（包括预训练的 Transformer 参数和新的线性层参数）。

- **数据与模型的形状变化**
    - **输入形状:** `[B, T_dec]`
    - **B:** 批量大小 (Batch Size)。
    - **T_dec:** 批次内最长序列的长度（短序列需要用 `pad` 填充）。
    - **Transformer 输出:** `[B, T_dec, 768]`
        - 模型为序列中的每一个 token 都输出了一个 768 维的特征向量。
    - **特征提取:** 从 `[B, T_dec, 768]` 中提取出 `Extract` token 对应的向量，得到 `[B, 768]`。
    - **最终输出 (Logits):** 将 `[B, 768]` 输入线性层，得到 `[B, Num_Classes]`。
        - `Num_Classes` 是任务的类别数量（例如，情感二分类就是 2）。
    - **标签 (Labels):** `[B]`
        - 一个一维张量，存放每个样本的正确类别索引（例如 `[0, 1, 1, 0, ...]`）。

- **特殊指令符说明：**
    - `Start`: 序列开始的信号。
    - `Delim` (Delimiter): 分隔符，用于隔开两个不同的文本部分。
    - `Extract`: 提取特征的信号点，标志着输入的结束。
    - **分类 (Classification):**
        - **结构:** `[Start] Text [Extract]`
        - **说明:** 将单个文本前后包裹起来。
    - **蕴含 (Entailment):**
        - **结构:** `[Start] Premise [Delim] Hypothesis [Extract]`
        - **说明:** 将前提和假设用分隔符拼接。
    - **相似度 (Similarity):**
        - **结构:** 需要构造两个输入序列，并将它们的 `Extract` 特征向量相加后再送入线性层。
        `[Start] Text 1 [Delim] Text 2 [Extract]`
        `[Start] Text 2 [Delim] Text 1 [Extract]`
        - **说明:** 这种方式可以消除句子顺序带来的影响。
    - **多选 (Multiple Choice):**
        - **结构:** 为每一个选项都构造一个独立的输入序列。
        `[Start] Context [Delim] Answer 1 [Extract]`
        `[Start] Context [Delim] Answer 2 [Extract]`
        ...
        - **说明:** 将所有序列分别通过模型得到各自的分数，最后通过 Softmax 归一化，选出概率最高的答案。

#### 损失函数
- **预训练损失：交叉熵损失函数**
![预训练损失函数](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/gpt1-2.png?raw=true)
在这个公式中，`k` 代表“上下文窗口的大小”，但在 GPT-1 的实际架构中，它并不是一个固定的、小的数字，而是指模型在预测当前词时，能够回看的所有前面词语，其最大值由模型的最大序列长度决定。

- **监督微调损失：**
![监督微调损失函数](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/gpt1-3.png?raw=true)

- **联合训练损失：**
![联合训练损失函数](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/gpt1-4.png?raw=true)

### GPT-2
[GPT-2论文原文](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

**架构细节：** 沿用了GPT-1的Transformer Decoder结构

| 模型 | 参数量 | 层数 (Layers) | 隐藏层维度 (d_model) |
| :--- | :--- | :--- | :--- |
| GPT-2 Small | 1.24 亿 (124M) | 12 | 768 |
| GPT-2 Medium | 3.55 亿 (355M) | 24 | 1024 |
| GPT-2 Large | 7.74 亿 (774M) | 36 | 1280 |
| GPT-2 XL | 15.58 亿 (1.5B) | 48 | 1600 |

GPT-1 和 GPT-2 在预训练的根本方法论上是统一的。其核心区别在于，GPT-2 通过将数据和模型的规模提升到前所未有的高度，使其“涌现”出了强大的零样本（Zero-Shot）能力，从而摆脱了对监督微调的依赖，开创了通过提示（Prompt）直接与模型交互解决任务的新范式。

### GPT-3
[GPT-3论文原文](https://arxiv.org/abs/2005.14165)
- **架构细节：**
| 模型 | 参数量 | 层数 (Layers) | 隐藏层维度 | 上下文窗口 (T_dec) |
| :--- | :--- | :--- | :--- | :--- |
| GPT-1 | 1.17 亿 | 12 | 768 | 512 |
| GPT-2 XL | 15 亿 | 48 | 1600 | 1024 |
| GPT-3 (Davinci) | 1750 亿 | 96 | 12288 | 2048 |

GPT-3的参数量是GPT-2最大模型的100多倍，上下文窗口再次翻倍，达到了2048个token。

- **In-Context Learning**
GPT-3的预训练流程在方法论上与GPT-1和GPT-2完全相同，但所有数字都达到了新的量级。
GPT-3不再仅仅讨论Zero-Shot，而是提出了一个包含三种主要模式的谱系，彻底摆脱了对微调的依赖：
    - **零样本 (Zero-Shot)：** 与GPT-2相同。只给模型任务的自然语言描述，不给任何示例。
      ```
      Translate English to French:
      cheese =>
      ```
    - **一样本 (One-Shot)：** 给模型任务描述，并额外提供一个任务示例。这对于人类来说是非常自然的教学方式。
      ```
      Translate English to French:
      sea otter => loutre de mer
      cheese => 
      ```
    - **少样本 (Few-Shot)：** 给模型任务描述，并提供几个（通常是10到100个）任务示例。这些示例全部被“塞”进模型的上下文窗口中，作为本次推理的条件。
      ```
      Translate English to French:
      sea otter => loutre de mer
      peppermint => menthe poivrée
      cheese => 
      ```
这个范式的核心是：模型的参数在推理时是完全冻结的。“学习”这个动作，是在模型的一次前向传播中，通过注意力机制“消化”和“理解”你提供的示例来完成的。这是一种高效的元学习 (Meta-Learning) 形式。

- **推理：In-Context Learning的具体实现**
GPT-3的推理完全基于Prompt，根据你提供的示例数量，分为Zero/One/Few-Shot。
    - 所有内容（任务描述、示例、最终问题）都被格式化成一个单一的字符串，然后Tokenize后作为模型的输入。
    - 这个输入序列的总长度不能超过2048个token。
    - **模型输入:** `[1, T_prompt]` (批量大小B通常为1，`T_prompt` <= 2048)
    - **模型输出:** 模型续写这个Prompt。它不知道自己正在“做任务”，它只是在基于你给出的、 highly structured 的上下文，生成概率最高的下一个token序列。
    - **标签:** 没有标签。推理阶段是纯粹的生成。
    - **两种停止方式：** 1、自然停止：模型自己预测生成“停止符” 2、强制停止：用户设定的最大长度

- **数据构造:**
    1. **在全局数据层面（拼接“文本香肠”时）：**
       为了让模型理解“文档”这个概念的边界，在把所有文本（来自WebText, Books等）拼接成一个巨大的、连续的数据流时，会在每两个独立的文档之间，插入一个特殊的`<|endoftext|>` token。
       所以，这根巨大的“文本香肠”的内部结构是这样的：
       `[文档A的文本...] <|endoftext|> [文档B的文本...] <|endoftext|> [文档C的文本...]`
       在这个层面上，`<|endoftext|>` 扮演的是文档之间的自然分隔符。
    2. **在构建训练样本层面（从“香肠”上切块时）：**
       模型训练时，是从上面那根长长的“香肠”中，随机地切下一个固定长度（如2048个token）的片段。
       这个切下来的片段，我们不会在它的开头和结尾再额外添加任何 `BOS` 或 `EOS` 符号。 这就是我之前强调的、与传统机器翻译模型不同的地方。
       但是，因为`<|endoftext|>`本身就存在于“香肠”内部，所以我们切下来的这个2048长度的片段，有可能会“恰好”包含一个或多个`<|endoftext|>` token。

### GPT-3-instruct
- [InstructGPT论文原文](https://arxiv.org/abs/2203.02155)
- **训练流程**
![InstructGPT 训练流程图](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/gpt-ins-1.png?raw=true)

#### 步骤 1: 监督微调 (Supervised Fine-Tuning, SFT)
这是训练的起点，目标是让预训练的GPT-3模型初步具备遵循指令的能力。
- **输入数据:**
一批由“提示”和人工编写的“示范回答”组成的数据对。
在模型内部，提示和示范回答会拼接成一个完整的序列。
**维度表示:** `(B, T_dec)`，其中 `B` 是批量大小（batch size），`T_dec` 是“提示”和“示范回答”拼接后的总长度。
- **训练过程中的维度变化:**
输入序列（维度 `(B, T_dec)`）被送入GPT-3模型。
模型对序列中的每一个token进行前向传播，输出一个在整个词汇表（Vocabulary）上的概率分布，用来预测下一个token。
输出的Logits维度为 `(B, T_dec, V)`，其中 `V` 是词汇表的大小。
- **模型输出 (Forward Pass Output):**
在每个时间步（token位置）上预测的下一个token的Logits。
- **标签 (Label):**
标签就是人工编写的“示范回答”本身。训练目标是最大化模型生成这个示范回答的概率。
这通过一个标准的自回归语言模型损失函数（交叉熵损失）来实现，即根据前面的tokens预测序列中的下一个token。

#### 步骤 2: 训练奖励模型 (Reward Model, RM)
训练一个模型，让它能够接收一个“提示（prompt）”和模型生成的“回答（response）”，然后输出一个单一的数值（标量奖励）。这个数值代表了人类标注员对这个回答质量的评分。

- **输入数据:**
输入数据来自标注员对模型输出的排序。例如，对于提示P，有回答A, B, C, D，标注员排序为 D > C > A = B。
这个排序会被拆解成多组成对的比较数据，例如 `(P, D) vs (P, C)`，`(P, D) vs (P, A)` 等。每一对包含一个更受偏好的回答 和一个不太受偏好的回答 。
模型会分别处理 `(P, y_w)` 和 `(P, y_l)` 这两个拼接后的序列。
**维度表示:** 对于每一个比较对（ `(P, D) vs (P, C)`），输入是两个序列，每个序列维度为 `(B, T_dec)`。

- **训练过程中的维度变化:**
将一个“提示-回答”序列（维度 `(B, T_dec)`）输入到一个修改版的SFT模型中。这个模型的最终词嵌入层被替换成一个线性层，用于输出一个标量值。
模型对整个序列进行前向传播。
最终输出一个单一的标量值，即奖励分数（reward score）。
输出维度为 `(B, 1)`。

- **模型输出 (Forward Pass Output):**
一个标量奖励分数 `r`。
- **标签 (Label):**
这里没有显式的数值标签。标签是“人类的偏好”，体现在损失函数中。
损失函数的目标是最大化“winner”回答的奖励分数与“loser”回答的奖励分数之间的差距。公式为：
`loss = -E [log(σ(r(P, y_w) - r(P, y_l)))]`
这个损失函数会驱动模型给人类更喜欢的回答打更高的分。

#### 步骤 3: 强化学习 (Reinforcement Learning, RL) 进行微调
这是最后一步，将奖励模型作为回报函数，利用PPO算法进一步微调SFT模型，使其生成的回答能获得更高的奖励分数。

- **输入数据:**
仅需要一个不带任何人工标注的提示数据集。
**维度表示:** `(B, T_prompt)`，其中 `T_prompt` 是提示的长度。

- **训练过程中的维度变化 (PPO 迭代过程):**
    - **生成 (Rollout):** 当前策略模型（Policy，由SFT模型初始化）接收提示 P（维度 `(B, T_prompt)`），并生成回答 y。拼接成完整序列维度为 `(B, T_dec=T_prompt+T_y)`。
    - **评估 (Evaluation):** 将生成的“提示-回答”对 `(P, y)` 输入到上一步训练好的、且权重固定的奖励模型（RM）中。RM输出一个标量奖励分数 `r`。
    - **策略更新 (Update):** PPO算法使用这个奖励分数 `r` 作为优化的信号，来更新策略模型（Policy）的参数。更新的目标是让策略模型生成能够获得更高奖励分数的回答。
    - 为了防止模型过分“迎合”奖励模型而偏离原始语言风格，PPO的目标函数中还加入了一项KL散度惩罚，用来约束当前策略模型和SFT模型的输出分布不要相差太大。

#### RLHF中的几点思考
##### 关于Reward Model

1.  **目标与模型结构**
    模型规模: 论文中提到，他们主要使用60亿（6B）参数的奖励模型，因为175B规模的模型在训练时不稳定，不适合在后续的强化学习阶段作为价值函数使用。

2.  **数据收集与处理**
    为了训练奖励模型，需要一个能够体现人类偏好的数据集。
    - **数据收集方式:** 为了提高效率，研究人员并非让标注员对两个回答进行简单的二选一。而是给标注员一个提示，然后展示由模型生成的 K 个 (K介于4到9之间) 不同的回答，并要求标注员对这 K 个回答进行排序（从最好到最差）。
    - **数据集转换:** 一个包含 K 个回答的排序，可以被分解成 `C(K, 2)` 个成对的比较数据。例如，如果一个标注员将4个回答排序为 D > C > B > A，那么就可以生成 `C(4, 2) = 6` 个比较对：`(D, C)`, `(D, B)`, `(D, A)`, `(C, B)`, `(C, A)`, `(B, A)`。在每个比较对中，都有一个更优的回答和一个较差的回答。

3.  **关键的训练策略：处理数据相关性**
    - **发现的问题:** 来自同一个提示的 `C(K, 2)` 个比较对是高度相关的。如果把所有这些比较对简单地混合（shuffle）到一个大数据集中，然后进行训练，模型会很快过拟合。
    - **解决方案:** 为了解决这个问题，研究人员采用了创新的批处理方式。他们将源自同一个提示的所有 `C(K, 2)` 个比较对视为一个单独的批处理元素（a single batch element）。
    - **该方案的优势:**
        - **防止过拟合:** 通过将相关的比较数据捆绑处理，避免了模型过分学习单个比较对的特性。
        - **计算效率高:** 对于一个提示的 K 个回答，只需要进行 K 次前向传播来计算每个回答的奖励分数。然后，利用这 K 个分数来计算所有 `C(K, 2)` 个比较对的损失。这远比为每个比较对都进行两次前向传播（总共 `2 * C(K, 2)` 次）要高效得多。

4.  **损失函数详解**
![奖励模型损失函数](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/gpt-ins-2.png?raw=true)

奖励模型的训练目标体现在其特定的损失函数中：
$$
loss(\theta) = - \frac{1}{C(K, 2)} E [log(\sigma(r_\theta(x, y_w) - r_\theta(x, y_l)))]
$$
让我们来分解这个公式：
- **$r_\theta(x, y)$:** 这是奖励模型的核心。它接收提示 x 和回答 y，并输出一个由模型参数 θ 决定的标量奖励分数。
- **$y_w$ 和 $y_l$:** 分别代表在一个比较对中，“获胜”（winner）的回答和“落败”（loser）的回答。
- **$r_\theta(x, y_w) - r_\theta(x, y_l)$:** 这是训练的关键部分，即计算“获胜者”和“落败者”的奖励分数之差。
- **$\sigma(...)$:** 这是 Sigmoid 函数。它将任意实数（分数差）映射到 (0, 1) 区间，可以被看作是模型认为 `yw` 比 `yl` 更好的概率。
- **`log(...)`:** 取对数似然。训练的目标是最大化这个概率。
- **`- E[...]`:** 通过在整个数据集 D 上取期望（平均值）并加上负号，我们将“最大化对数似然”问题转化为了一个标准的“最小化损失”问题，这正是神经网络训练所做的。

通俗地讲，这个损失函数的目标是：
尽可能地让模型给人类偏好的回答（`yw`）打出比不偏好的回答（`yl`）更高的分数，并且分数差越大越好。
通过最小化这个损失函数，奖励模型就学会了如何像人类标注员一样，对不同回答的质量进行打分。

##### Reward Model中的数据处理
- **方案一：简单混合 (Shuffle) 训练 (会导致过拟合)**
    - **输入 (Input)**
        - **数据准备：**
          我们将6个比较对打散，随机放入一个大训练集中。假设在一个批次（Batch）中，有32个比较对
          - 比较对1: 来自提示P1
          - 比较对2: 来自提示P7
          - 比较对3: 来自提示P54
          - ...
          - 比较对32: 来自提示P2
          这个批次包含 `32 * 2 = 64` 个序列（每个比较对有winner和loser两个序列），例如比较对1为:`[p1,yw],[p1,yl]`
    - **输入到模型：**
        - **维度:** `(64, T_dec)`
        - **64:** 代表这个批次包含64个序列（32个winner序列，32个loser序列）。
        - **T_dec:** 代表“提示+回答”拼接后的token总长度。
    - **模型输出 (Process & Output)**
        - `[64,]`得到64个分数，计算32个独立的损失值，最后求平均并更新模型。相当于需要`2 * C(K, 2)` 次计算。

- **方案二：捆绑处理 (Bundled Processing) (论文采用的方案)**
    - **输入 (Input)**
        - **数据准备：**
          假设在一个批次（Batch）中，有32个提示，每个提示 有 K=4 个回答 A, B, C, D。
    - **输入到模型：**
        - 输入元素1: 提示 P 1和回答 A 拼接成的序列 `[P, A]`。
        - 输入元素2: 提示 P 1和回答 B 拼接成的序列 `[P, B]`。
        - 输入元素3: 提示 P 1和回答 C 拼接成的序列 `[P, C]`。
        - 输入元素4: 提示 P 2和回答 D 拼接成的序列 `[P, D]`。
        - ...一共`32*K`个序列
    - **维度与含义：**
        - **维度:** `(32*K, T_dec)`
        - **32*K:** 代表这个批次包含与（提示，回答）序列。
        - **T_dec:** 拼接后序列的长度。
    - **模型输出结果：**
        - 输出1: `[P, A] -> score_A`
        - 输出2: `[P, B] -> score_B`
        - 输出3: `[P, C] -> score_C`
        - 输出4: `[P, D] -> score_D`
        - ...
        - `[32*k,]`得到32*k个分数，计算32*`C(K, 2)`个损失值，最后求平均并更新模型。相当于需要K次计算。
    - **损失计算 (Loss Calculation):**
        - **步骤a: 构建比较对:** 利用输出的4个分数，在计算图内部（in-memory）构建出所有6个比较对的得分差：
          `score_D - score_C`
          `score_D - score_B`
          ...
          `score_B - score_A`
        - **步骤b: 计算每个对的损失:**
          `loss_D_C = -log(σ(score_D - score_C))`
          `loss_D_B = -log(σ(score_D - score_B))`
          ...
          `loss_B_A = -log(σ(score_B - score_A))`
        - **步骤c: 计算总平均损失:**
          `total_loss = (loss_D_C + loss_D_B + ... + loss_B_A) / 6`
          这个平均操作就是 `(1 / C(K, 2))` 的作用。

##### RLHF的目标函数
![RLHF 目标函数](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/gpt-ins-3.png?raw=true)

这是一个目标函数 (Objective Function)，我们的目的是让它的值最大化 (Maximize)。
`π_φ^RL(x)`是当前的policy模型。

- **第一部分：追求高分 (The Reward Term)**
`r_θ(x, y)`
    - **这是什么？**
      `r_θ(x, y)` 是奖励模型 (RM) 给出的分数。
      `x` 是输入的提示 (prompt)，`y` 是当前策略模型 `π_φ^RL` 生成的回答 (response)。
      这个分数代表了我们训练好的“人类品味代理裁判”对当前回答的满意度。分数越高，说明这个回答越符合人类的偏好。
    - **意义与目标：**
      这是最主要的驱动力。 通过最大化这一项，策略模型 `π_φ^RL` 会被激励去生成那些能够“讨好”奖励模型的回答。
      通俗地说，这是在告诉模型：“去生成那些能得高分的答案！”

- **第二部分：保持本真，防止“走火入魔” (The KL Penalty Term) 惩罚项**
`- β log(π_φ^RL(y|x) / π^SFT(y|x))`
    - **这是什么？**
        - `π_φ^RL(y|x)`: 当前正在训练的模型，生成回答 `y` 的概率。
        - `π^SFT(y|x)`: 固定不动的、第一阶段训练好的监督微调模型（SFT model），生成回答 `y` 的概率。
        - `log(π_RL / π_SFT)`: 这是两个概率分布之间KL散度的一种形式。它衡量了当前模型 `π_RL` 相对于初始模型 `π_SFT` 的“偏离程度”。
        - `- β`: `β` 是一个控制强度的系数，前面的负号至关重要。
    - **意义与目标：**
      这是一个惩罚项/约束项。 我们的目标是最大化整个函数，而这一项带了负号，所以模型会试图让 `log(...)` 这一部分变得尽可能小。
      `log(...)` 什么时候最小？当 `π_RL` 和 `π_SFT` 非常接近时，比率接近1，`log(1)=0`，达到最小值。如果 `π_RL` 生成了一个 `π_SFT` 认为完全不可能生成的回答（即 `π_SFT(y|x)` 极小），那么这个比率会非常大，`log(...)` 也会非常大，导致惩罚变大，总目标函数减小。
      通俗地说，这是在告诉模型：“你可以去追求高分，但你生成的内容不能太离谱，必须和你最初从人类范例中学到的风格保持一致，不许为了分数而胡言乱语！”
      这有效防止了模型为了利用奖励模型的漏洞而生成一些奇怪、重复、不自然的文本（即过度优化）。

- **第三部分：温故知新，保持博学 (The Pretraining Term) 相当于交叉熵损失无负号**
`+ γE_x~D_pretrain [log(π_φ^RL(x))]`
    - **这是什么？**
        - `D_pretrain`: 这是GPT-3原始的、巨大的预训练数据集（例如，来自互联网的海量通用文本）。注意，这里的 `x` 是指一段预训练数据，而不是一个指令提示。
        - `log(π_φ^RL(x))`: 这是标准的语言模型训练目标。它衡量了当前模型 `π_φ^RL` 对于一段通用文本的理解和生成能力（即预测下一个词的能力）。
        - `γ`: 一个控制强度的系数。
    - **意义与目标：**
      这是一个“知识补充”项。 在模型专注于学习如何遵循人类指令（来自第一、二部分）的同时，我们还让它“分心”去温习一下它在预训练阶段学到的海量通用知识。
      通俗地说，这是在告诉模型：“在你努力成为一个听话的‘专才’（遵循指令）时，别忘了你曾经是一个无所不知的‘通才’（通用语言模型）。要不断温习，保持你的知识广度！”
      这一项的目的是解决**“对齐税” (alignment tax)** 的问题。只进行前两部分的RL训练，可能会让模型在某些通用能力上（如常识问答、代码理解等）发生退化。加入这一项，可以让模型在对齐的同时，保持其强大的通用能力。
