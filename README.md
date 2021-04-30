# Deep Recommenders
[![python](https://img.shields.io/badge/python-3.7-brightgreen)](requirements.txt)
[![tensorflow](https://img.shields.io/badge/tensorflow-2.3-brightgreen)](requirements.txt)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![test](https://img.shields.io/badge/test-passing-brightgreen)](TEST)


***Deep Recommenders***主要用于自我学习和提升, 还希望能够帮助对推荐系统感兴趣的朋友和同学，共同进步～

由于本人水平有限，如有错误，还望指正～

框架参考：[TensorFlow Recommenders](https://github.com/tensorflow/recommenders)

## Experiments

### Ranking

- **FM**, [
        <font size=1.5>
            *Factorization Machines, Osaka, 2010*
        </font>](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
        [Code](deep_recommenders/layers/fm.py) Expr
- **WDL**, [
        <font size=1.5>
            *Wide & Deep Learning for Recommender Systems, Google, DLRS, 2016*
        </font>](https://arxiv.org/abs/1606.07792)
        Code Expr
- **PNN**, [
        <font size=1.5>
            *Product-based Neural Networks for User Response Prediction, IEEE, 2016*
        </font>](https://arxiv.org/abs/1611.00144)
        Code Expr
- **FNN**, [
        <font size=1.5>
            *Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction, RayCloud, ECIR, 2016*
        </font>](https://arxiv.org/abs/1601.02376)
        Code Expr
- **DeepFM**, [
        <font size=1.5>
            *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, Huawei, IJCAI, 2017*
        </font>](https://www.ijcai.org/proceedings/2017/0239.pdf)
        Code [Expr](experiments/deepfm.ipynb)
- **DCN**, [
        <font size=1.5>
            *Deep & Cross Network for Ad Click Predictions, Google, KDD, 2017*
        </font>](https://arxiv.org/abs/1708.05123)
        [Code](deep_recommenders/layers/dcn.py) Expr
- **xDeepFM**, [
        <font size=1.5>
            *xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, Microsoft, KDD, 2018*
        </font>](https://arxiv.org/pdf/1803.05170.pdf)
        [Code](deep_recommenders/layers/xdeepfm.py) Expr
- **DIN**, [
        <font size=1.5>
            *Deep Interest Network for Click-Through Rate Prediction, Alibaba, KDD, 2018*
        </font>](https://arxiv.org/abs/1706.06978)
        [Code](deep_recommenders/layers/din.py) Expr
- **DIEN**, [
        <font size=1.5>
            *Deep Interest Evolution Network for Click-Through Rate Prediction, Alibaba, AAAI, 2019*
        </font>](https://arxiv.org/abs/1809.03672)
        Code Expr
- **DLRM**, [
        <font size=1.5>
            *Deep Learning Recommendation Model for Personalization and Recommendation Systems, Facebook, 2019*
        </font>](https://arxiv.org/abs/1906.00091)
        Code Expr

### Retrieval

- **DSSM**, [
        <font size=1.5>
            *Learning Deep Structured Semantic Models for Web Search using Clickthrough Data, Microsoft, CIKM, 2013*
        </font>](https://dl.acm.org/doi/10.1145/2505515.2505665)
        Code Expr
- **YoutubeNet**, [
        <font size=1.5>
            *Deep Neural Networks for YouTube Recommendations, Google, RecSys, 2016*
        </font>](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)
        Code Expr
- **SBC**, [
        <font size=1.5>
            *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations, Google, RecSys, 2019*
        </font>](https://dl.acm.org/doi/10.1145/3298689.3346996)
        [Code](deep_recommenders/layers/factorized_top_k.py) 
        [Expr](experiments/deep_retrieval.ipynb)
- **MOBIUS**, [
        <font size=1.5>
            *MOBIUS: Towards the Next Generation of Query-Ad Matching in Baidu’s Sponsored Search, Baidu, KDD, 2019*
        </font>](http://research.baidu.com/Public/uploads/5d12eca098d40.pdf)
        Code Expr
- **EBR**, [
        <font size=1.5>
            *Embedding-based Retrieval in Facebook Search, Facebook, KDD, 2020*
        </font>](https://arxiv.org/abs/2006.11632)
        Code Expr
- **Item2Vec**, [
        <font size=1.5>
            *Item2Vec: Neural Item Embedding for Collaborative Filtering, Microsoft, MLSP, 2016*
        </font>](https://arxiv.org/vc/arxiv/papers/1603/1603.04259v2.pdf)
        Code Expr
- **Airbnb**, [
        <font size=1.5>
            *Real-time Personalization using Embeddings for Search Ranking at Airbnb, Airbnb, KDD, 2018*
        </font>](https://dl.acm.org/doi/10.1145/3219819.3219885)
        Code Expr
- **DeepWalk**, [
        <font size=1.5>
            *DeepWalk: Online Learning of Social Representations, StonyBrook, KDD, 2014*
        </font>](https://arxiv.org/abs/1403.6652)
        Code Expr
- **EGES**, [
        <font size=1.5>
            *Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba, Alibaba, KDD, 2018*
        </font>](https://arxiv.org/abs/1803.02349)
        Code Expr
- **GCN**, [
        <font size=1.5>
            *Semi-Supervised Classification with Graph Convolutional Networks, ICLR, 2017*
        </font>](https://arxiv.org/abs/1609.02907)
        [Code](deep_recommenders/layers/gnn.py#L16) Expr
- **GraphSAGE**, [
        <font size=1.5>
            *Inductive Representation Learning on Large Graphs, NIPS, 2017*
        </font>](https://arxiv.org/abs/1706.02216)
        Code Expr
- **PinSage**, [
        <font size=1.5>
            *Graph Convolutional Neural Networks for Web-Scale Recommender Systems, Pinterest, KDD, 2018*
        </font>](https://arxiv.org/abs/1806.01973)
        Code Expr
- **IntentGC**, [
        <font size=1.5>
            *IntentGC: a Scalable Graph Convolution Framework Fusing Heterogeneous Information for Recommendation, Alibaba, KDD, 2019*
        </font>](https://arxiv.org/abs/1907.12377)
        Code Expr
- **GraphTR**, [
        <font size=1.5>
            *Graph Neural Network for Tag Ranking in Tag-enhanced Video Recommendation, Tencent, CIKM, 2020*
        </font>](https://dl.acm.org/doi/abs/10.1145/3340531.3416021)
        Code Expr
    
### NLP

- **Transformer**, [
        <font size=1.5>
            *Attention Is All You Need, Google, NeurlPS, 2017*
        </font>](https://arxiv.org/abs/1706.03762)
        [Code](deep_recommenders/layers/nlp/transformer.py) 
        [Expr](experiments/transformer.ipynb)



