---
license: apache-2.0
task_categories:
- question-answering
- text-generation
language:
- zh
- ii
size_categories:
- 100K<n<1M
---

# Nuosu-Benchmark 总结说明

`Nuosu-Benchmark` 是面向极低资源语言（规范彝文）的标准化评估基准，旨在系统评测大语言模型在跨语言理解、语义对齐与推理生成方面的综合能力。  

---

## 数据集总体构成

`Nuosu-Benchmark` 由 5 个子数据集组成，共 **110,513** 个高质量样本，覆盖词汇、句法、语义、语用到综合推理等多层级能力。

| 数据集 | 样本数 | 语言层次 | 任务类型 | 核心测试能力 |
|---|---:|---|---|---|
| NuosuLex | 13,581 | 词汇层、语义层 | 词汇翻译、词义辨析 | 词义映射、词汇对齐 |
| NuosuSen | 3,584 | 句法层、语义层 | 句子翻译、口语表达 | 语法结构、日常语用 |
| NuosuEpic | 17,644 | 语用层 | 诗歌翻译、文化释意 | 跨语理解、文化语境 |
| NuosuGov | 1,712 | 语用层 | 公文翻译、阅读理解 | 抽象语义、术语一致性 |
| NuosuEdu | 73,992 | 综合层 | 选择、判断、问答 | 语言逻辑、推理逻辑 |

# Nuosu-Benchmark Overview

**Nuosu-Benchmark** is a standardized evaluation benchmark designed for extremely low-resource languages (specifically Standard Yi / Nuosu). It aims to systematically assess large language models in cross-lingual understanding, semantic alignment, and reasoning generation.

---

## Dataset Composition

Nuosu-Benchmark consists of **5 sub-datasets**, totaling **110,513** high-quality samples. It covers multiple linguistic levels, including lexical, syntactic, semantic, pragmatic, and comprehensive reasoning capabilities.

| Dataset   | Samples | Linguistic Level        | Task Type                          | Core Evaluation Focus                  |
|-----------|--------:|------------------------|------------------------------------|----------------------------------------|
| NuosuLex  | 13,581  | Lexical, Semantic      | Word translation, sense disambiguation | Lexical mapping, semantic alignment    |
| NuosuSen  | 3,584   | Syntactic, Semantic    | Sentence translation, colloquial expression | Grammar structure, everyday pragmatics |
| NuosuEpic | 17,644  | Pragmatic              | Poetry translation, cultural interpretation | Cross-lingual understanding, cultural context |
| NuosuGov  | 1,712   | Pragmatic              | Official document translation, reading comprehension | Abstract semantics, terminology consistency |
| NuosuEdu  | 73,992  | Comprehensive          | Multiple choice, classification, QA | Linguistic reasoning, logical inference |
