### Head-Lexicalized Bidirectional Tree LSTMs 
This repo contains code implementations for my TACL paper [Head-Lexicalized Bidirectional Tree LSTMs](https://www.aclweb.org/anthology/Q17-1012) and the previous arxiv version [Bidirectional Tree-Structured LSTM with Head Lexicalization](https://arxiv.org/abs/1611.06788).   

This code was written by using the framework [cnn-v1](!https://github.com/clab/cnn-v1). 

### Models
This repo contains codes for the following four models: 

+ **bidir+lex**. This is the full model reported in the paper, namely with bottom-up information flows, top-down information flows and head lexicalizations. 
+ **bottomup+lex**. This model contains bottom-up information flows and head lexicalizations. 
+ **topdown+lex**.  This model contains top-down information flows and head lexicalizations. 
+ **bottomup**.  This is a basic bottom-up tree-structured LSTMs. 

To compile each model, please ``cd`` to the corresponding fold name and check the ``compile.md``. All the compiling follows the same logics. You may need to install appropriate ``boost`` and ``eigen`` libraries. ``Boost 1.59.0`` and eigen releases around Feb, 2016 are recommended options. 

### How to do training and testing?

Please check the `train-XXX.sh` and `test-XXX.sh` in the `exp` folder, where `XXX` corresponds to the names of the four kinds of models. 

### Data

Please check the ``data`` folder for the training resouces and the pretrained embeddings I used. 

### Pretrained Models 

For the model achieved the best score on the fine-grained root level sentiment classification, you can download it via the [Google drive link](https://drive.google.com/open?id=1r2PfhyStghi2kr7eSmnN_zTXHWRzJy5r). 

### Citations 

```latex
@article{Q17-1012,
    title = "Head-Lexicalized Bidirectional Tree LSTMs",
    author = "Teng, Zhiyang  and
      Zhang, Yue",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "5",
    number = "1",
    year = "2017",
    url = "https://www.aclweb.org/anthology/Q17-1012",
    pages = "163--177",
    abstract = "Sequential LSTMs have been extended to model tree structures, giving competitive results for a number of tasks. Existing methods model constituent trees by bottom-up combinations of constituent nodes, making direct use of input word information only for leaf nodes. This is different from sequential LSTMs, which contain references to input words for each node. In this paper, we propose a method for automatic head-lexicalization for tree-structure LSTMs, propagating head words from leaf nodes to every constituent node. In addition, enabled by head lexicalization, we build a tree LSTM in the top-down direction, which corresponds to bidirectional sequential LSTMs in structure. Experiments show that both extensions give better representations of tree structures. Our final model gives the best results on the Stanford Sentiment Treebank and highly competitive results on the TREC question type classification task.",
}
```