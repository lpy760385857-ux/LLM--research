# Word2Vec 作者源码

对应论文：[Efficient Estimation of Word Representations in Vector Space](../01_Word2Vec_2013.pdf)。

初学者先读：[Word2Vec 源码阅读指南](Word2Vec源码阅读指南.md)，按训练流程理解变量、函数、CBOW、Skip-gram 和负采样。

## 源码来源

- 论文第 11 页脚注 4 指向原项目：<https://code.google.com/p/word2vec/>。
- 作者 Tomas Mikolov 的 GitHub 源码仓库：<https://github.com/tmikolov/word2vec>。
- 收录提交：`20c129af10659f7c50e86e3be406df663beff438`。
- 许可证：Apache License 2.0，原始 [LICENSE](word2vec/LICENSE) 与版权声明已保留。

本目录收录完整的 17 个上游受版本控制文件，逐文件核对内容与执行权限；代码未修改。来源记录见 [word2vec-source.json](word2vec-source.json)。源码直接纳入当前仓库，下载当前仓库即可获得，无需额外初始化子模块。上游 Git 历史留在作者仓库，编译产物不提交。

这是作者项目的后续版本，包含 CBOW、Skip-gram、层次 softmax、负采样和短语工具；并非严格冻结在首篇论文发表时的代码。负采样可结合 [后续论文](../04_Word2Vec_Negative_Sampling_2013.pdf) 阅读。

## 从哪里开始读

- [README.txt](word2vec/README.txt)：上游使用说明。
- [word2vec.c](word2vec/word2vec.c)：主要训练实现。先看 `main` 的参数读取，再看 `TrainModel` 和 `TrainModelThread`；`if (cbow)` 分支对应 CBOW，另一分支对应 Skip-gram。
- [makefile](word2vec/makefile)：编译训练程序、短语工具、相似词查询和类比评估工具。
- [word2phrase.c](word2vec/word2phrase.c)：短语处理。
- [distance.c](word2vec/distance.c) 与 [word-analogy.c](word2vec/word-analogy.c)：相似词与类比查询。

## 编译与小语料实践

需要 Linux、GCC 和 Make。以下命令从本仓库根目录执行：

```bash
make -C 基础/代码/word2vec -j2
mkdir -p /tmp/word2vec-study
python3 - <<'PY'
from pathlib import Path
text = (
    'king queen man woman\n'
    'king man prince boy\n'
    'queen woman princess girl\n'
    'cat dog animal pet\n'
    'apple orange fruit food\n'
    'man woman boy girl\n'
)
Path('/tmp/word2vec-study/corpus.txt').write_text(text * 1000)
PY

# CBOW：用上下文预测中心词。
基础/代码/word2vec/word2vec -train /tmp/word2vec-study/corpus.txt -output /tmp/word2vec-study/cbow.txt -size 20 -window 3 -min-count 1 -sample 0 -negative 5 -hs 0 -threads 1 -binary 0 -cbow 1 -iter 5 -debug 0

# Skip-gram：用中心词预测上下文。
基础/代码/word2vec/word2vec -train /tmp/word2vec-study/corpus.txt -output /tmp/word2vec-study/skipgram.txt -size 20 -window 3 -min-count 1 -sample 0 -negative 5 -hs 0 -threads 1 -binary 0 -cbow 0 -iter 5 -debug 0
```

已验证：五个 C 工具均可编译；两种训练模式均成功输出 17 个 20 维向量，输出行数、维度与数值有效性检查通过。小语料用于验证训练流程，不能据此判断真实语义表现或复现论文指标。

训练结果位于 `/tmp/word2vec-study/`。上游示例脚本使用额外语料和旧下载地址；上述小语料流程不依赖这些下载。
