# 基础：词向量经典论文

核心阅读顺序：**Word2Vec → GloVe → FastText**。学习 Word2Vec 的训练细节时，可穿插阅读负采样补充论文。

Word2Vec 配套代码已收录在 [代码/word2vec](代码/word2vec)，来源为作者 Tomas Mikolov 的 GitHub 仓库。参见 [源码来源与运行说明](代码/README.md)。

## 论文目录

| 类型 | 论文 PDF | 年份 | 阅读重点 | 官方入口 |
| --- | --- | --- | --- | --- |
| 核心 1 | [Efficient Estimation of Word Representations in Vector Space](01_Word2Vec_2013.pdf) | 2013 | CBOW、Skip-gram、词向量与词类比 | [arXiv](https://arxiv.org/abs/1301.3781) |
| 核心 2 | [GloVe: Global Vectors for Word Representation](02_GloVe_2014.pdf) | 2014 | 全局共现统计、共现概率比、加权最小二乘目标 | [ACL Anthology](https://aclanthology.org/D14-1162/) |
| 核心 3 | [Enriching Word Vectors with Subword Information](03_FastText_2017.pdf) | 2017 | character n-gram、子词向量、OOV 与低频词 | [ACL Anthology](https://aclanthology.org/Q17-1010/) |
| 补充 | [Distributed Representations of Words and Phrases and their Compositionality](04_Word2Vec_Negative_Sampling_2013.pdf) | 2013 | 负采样、高频词下采样、短语表示 | [arXiv](https://arxiv.org/abs/1310.4546) |

## 作者与版本

- **Word2Vec**：Tomas Mikolov、Kai Chen、Greg Corrado、Jeffrey Dean。保存的是 arXiv:1301.3781v3，2013 年 9 月 7 日版本，共 12 页。
- **GloVe**：Jeffrey Pennington、Richard Socher、Christopher D. Manning。EMNLP 2014 论文；保存的是作者版 PDF，共 12 页。
- **FastText**：Piotr Bojanowski、Edouard Grave、Armand Joulin、Tomas Mikolov。发表于 TACL 2017，预印本最初发表于 2016 年；保存的是 arXiv:1607.04606v2，2017 年 6 月 19 日更新版，共 12 页。
- **Word2Vec 补充**：Tomas Mikolov、Ilya Sutskever、Kai Chen、Greg Corrado、Jeffrey Dean。保存的是 arXiv:1310.4546v1，2013 年 10 月 16 日版本，共 9 页。

## 阅读时回答这些问题

1. **Word2Vec**：CBOW 和 Skip-gram 分别输入什么、预测什么？训练出的哪组参数成为词向量？
2. **GloVe**：共现矩阵如何构造？为什么使用共现概率比和对数共现次数？它与 Word2Vec 的训练目标有什么不同？
3. **FastText**：一个词会被拆成哪些字符 n-gram？词向量如何组合？没见过的词如何得到向量？为什么低频词能受益？
4. **负采样补充**：为什么需要负采样？负样本从哪里来？高频词下采样解决什么问题？

实践建议：用同一小语料比较词相似度，并观察 FastText 对低频词和词表外词的表示；不要把产生了向量等同于理解了词义。

## 文件来源

PDF 从公开 GitHub 论文镜像获取，已核对标题、作者、正文及文件完整性，并保留上表中的官方入口。GloVe 的两个镜像版本正文文本一致；Word2Vec 的两个镜像副本 SHA-256 一致。FastText 采用 2017 年更新版。

每个文件的下载来源、版本、页数、字节数及 SHA-256 记录在 [sources.json](sources.json) 中。论文署名与原始内容均保留。
