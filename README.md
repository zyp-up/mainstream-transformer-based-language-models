
# 基于Transformer的三大主流语言模型架构

![思维导图概览](https://github.com/zyp-up/mainstream-transformer-based-language-models/blob/main/assets/%E5%9F%BA%E4%BA%8Etransformer%E7%9A%84%E4%B8%89%E5%A4%A7%E4%B8%BB%E6%B5%81%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84.png)

## 概述

Transformer模型自2017年被提出以来，已成为自然语言处理（NLP）领域的基石。其独特的自注意力机制（Self-Attention）使其能够高效地处理长距离依赖关系。基于其核心架构，衍生出了三种主流的语言模型范式：**Encoder-only**、**Decoder-only** 和 **Encoder-Decoder**。

- **Encoder-only (代表: BERT)**：采用Transformer的编码器部分，因其核心机制（双向自注意力）和核心训练目标（掩码语言模型）都是为了一个目标而设计的：一次性地、全局性地捕捉并编码整段文本的深层上下文信息，故天然适合于理解型任务，如文本分类、实体识别、情感分析等。其通过上下文双向编码来理解文本。

- **Decoder-only (代表: GPT系列)**：采用Transformer的解码器部分，专注于生成式任务，如文本生成、对话系统、文章撰写等。其通过自回归（Auto-regressive）的方式，根据上文生成下一个词。

- **Encoder-Decoder (代表: T5, BART)**：采用完整的Transformer架构，包含编码器和解码器。核心思想就是将几乎所有NLP的下游任务都统一为一种“文本到文本”（Text-to-Text）的格式。常用于序列到序列（Seq2Seq）任务，如机器翻译、文本摘要等。

