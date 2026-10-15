# Word2Vec Python 版本：从这里开始

源码：[word2vec-python](word2vec-python)。上游项目：[OlgaChernytska/word2vec-pytorch](https://github.com/OlgaChernytska/word2vec-pytorch)，收录提交 `18d4f24c4d82dde76de9f6e4d88bbcbcf68befc2`。

这是使用 PyTorch 的教学实现，包括 CBOW 和 Skip-gram，采用普通全词表 softmax 与交叉熵训练。它不是原版 C 代码的逐行翻译，也没有实现负采样或层次 softmax。

## 建议阅读顺序

1. [utils/model.py](word2vec-python/utils/model.py)：先看 `CBOW_Model.forward` 的三步：查上下文向量 → 求平均 → 输出候选词分数；再看 `SkipGram_Model`。
2. [utils/constants.py](word2vec-python/utils/constants.py)：查看向量维度、上下文长度和词频阈值。
3. [utils/dataloader.py](word2vec-python/utils/dataloader.py)：了解文本如何变成输入词 ID 和目标标签。
4. [utils/trainer.py](word2vec-python/utils/trainer.py)：看 `zero_grad → forward → loss → backward → step` 如何组成训练。
5. [train.py](word2vec-python/train.py) 与 [config.yaml](word2vec-python/config.yaml)：查看训练入口与参数配置。

## 与已收录 C 版对应

| C 版概念 | Python 版位置 |
| --- | --- |
| `syn0` 输入词向量表 | `self.embeddings` |
| 用词 ID 定位一行向量 | `self.embeddings(inputs_)` |
| CBOW 的 `neu1` 上下文平均 | `x.mean(axis=1)` |
| 输出侧参数 | `self.linear`，含权重和偏置，结构并非与 C 版逐项相同 |
| 手写更新系数与梯度 | PyTorch 的 `loss.backward()` 与 `optimizer.step()` |

`forward` 返回候选词的原始分数（logits）；`nn.CrossEntropyLoss` 内部完成相应的归一化与损失计算。不要把原始分数直接当成概率。

## 文件与运行状态

源码、配置、示意图、Notebook 和上游 README 均已保留；预训练权重与实验输出未随源码收录。Notebook 中涉及预训练文件的步骤需要另外准备权重。

已逐文件核对上游内容，所有 Python 文件语法解析通过。当前环境未安装 PyTorch / torchtext，因此没有运行完整训练，也未验证当前 Python 版本的依赖兼容性。

上游固定的是较旧依赖：`torch==1.9.0`、`torchtext==0.10.0`、`numpy==1.19.5`；运行前需要匹配的独立环境。`train.py` 还导入了 `yaml`，需要 PyYAML；上游 requirements 未列出该项。保留原依赖文件，便于核对来源。

上游 [README](word2vec-python/README.md) 声明 MIT 许可；该提交没有独立 LICENSE 文件。来源及收录范围见 [word2vec-python-source.json](word2vec-python-source.json)。
