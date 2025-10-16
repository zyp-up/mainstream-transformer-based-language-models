
# 基于Transformer的三大主流语言模型架构

![思维导图概览](https://storage.googleapis.com/static.aiforpro.com/1689264930263-12826/orignal_033d45c8188ea6891398bb89280d0752.png)

## 概述

Transformer模型自2017年被提出以来，已成为自然语言处理（NLP）领域的基石。其独特的自注意力机制（Self-Attention）使其能够高效地处理长距离依赖关系。基于其核心架构，衍生出了三种主流的语言模型范式：**Encoder-only**、**Decoder-only** 和 **Encoder-Decoder**。

- **Encoder-only (代表: BERT)**：采用Transformer的编码器部分，因其核心机制（双向自注意力）和核心训练目标（掩码语言模型）都是为了一个目标而设计的：一次性地、全局性地捕捉并编码整段文本的深层上下文信息，故天然适合于理解型任务，如文本分类、实体识别、情感分析等。其通过上下文双向编码来理解文本。

- **Decoder-only (代表: GPT系列)**：采用Transformer的解码器部分，专注于生成式任务，如文本生成、对话系统、文章撰写等。其通过自回归（Auto-regressive）的方式，根据上文生成下一个词。

- **Encoder-Decoder (代表: T5, BART)**：采用完整的Transformer架构，包含编码器和解码器。核心思想就是将几乎所有NLP的下游任务都统一为一种“文本到文本”（Text-to-Text）的格式。常用于序列到序列（Seq2Seq）任务，如机器翻译、文本摘要等。

