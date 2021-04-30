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

- **FM**
        [<sub>
            *Factorization Machines, Osaka, 2010*
        </sub>](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
        [Code](deep_recommenders/layers/fm.py) Expr
- **WDL**
        [<sub>
            *Wide & Deep Learning for Recommender Systems, Google, DLRS, 2016*
        </sub>](https://arxiv.org/abs/1606.07792)
        Code Expr
- **PNN**
        [<sub>
            *Product-based Neural Networks for User Response Prediction, IEEE, 2016*
        </sub>](https://arxiv.org/abs/1611.00144)
        Code Expr
- **FNN**
        [<sub>
            *Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction, RayCloud, ECIR, 2016*
        </sub>](https://arxiv.org/abs/1601.02376)
        Code Expr
- **DeepFM**
        [<sub>
            *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, Huawei, IJCAI, 2017*
        </sub>](https://www.ijcai.org/proceedings/2017/0239.pdf)
        Code [Expr](experiments/deepfm.ipynb)
- **DCN**
        [<sub>
            *Deep & Cross Network for Ad Click Predictions, Google, KDD, 2017*
        </sub>](https://arxiv.org/abs/1708.05123)
        [Code](deep_recommenders/layers/dcn.py) Expr
- **xDeepFM**
        [<sub>
            *xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, Microsoft, KDD, 2018*
        </sub>](https://arxiv.org/pdf/1803.05170.pdf)
        [Code](deep_recommenders/layers/xdeepfm.py) Expr
- **DIN**
        [<sub>
            *Deep Interest Network for Click-Through Rate Prediction, Alibaba, KDD, 2018*
        </sub>](https://arxiv.org/abs/1706.06978)
        [Code](deep_recommenders/layers/din.py) Expr
- **DIEN**
        [<sub>
            *Deep Interest Evolution Network for Click-Through Rate Prediction, Alibaba, AAAI, 2019*
        </sub>](https://arxiv.org/abs/1809.03672)
        Code Expr
- **DLRM**
        [<sub>
            *Deep Learning Recommendation Model for Personalization and Recommendation Systems, Facebook, 2019*
        </sub>](https://arxiv.org/abs/1906.00091)
        Code Expr

### Retrieval

- **DSSM**
        [<sub>
            *Learning Deep Structured Semantic Models for Web Search using Clickthrough Data, Microsoft, CIKM, 2013*
        </sub>](https://dl.acm.org/doi/10.1145/2505515.2505665)
        Code Expr
- **YoutubeNet**
        [<sub>
            *Deep Neural Networks for YouTube Recommendations, Google, RecSys, 2016*
        </sub>](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)
        Code Expr
- **SBC**
        [<sub>
            *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations, Google, RecSys, 2019*
        </sub>](https://dl.acm.org/doi/10.1145/3298689.3346996)
        [Code](deep_recommenders/layers/factorized_top_k.py) 
        [Expr](experiments/deep_retrieval.ipynb)
- **EBR**
        [<sub>
            *Embedding-based Retrieval in Facebook Search, Facebook, KDD, 2020*
        </sub>](https://arxiv.org/abs/2006.11632)
        Code Expr
- **Item2Vec**
        [<sub>
            *Item2Vec: Neural Item Embedding for Collaborative Filtering, Microsoft, MLSP, 2016*
        </sub>](https://arxiv.org/vc/arxiv/papers/1603/1603.04259v2.pdf)
        Code Expr
- **Airbnb**
        [<sub>
            *Real-time Personalization using Embeddings for Search Ranking at Airbnb, Airbnb, KDD, 2018*
        </sub>](https://dl.acm.org/doi/10.1145/3219819.3219885)
        Code Expr
- **DeepWalk**
        [<sub>
            *DeepWalk: Online Learning of Social Representations, StonyBrook, KDD, 2014*
        </sub>](https://arxiv.org/abs/1403.6652)
        Code Expr
- **EGES**
        [<sub>
            *Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba, Alibaba, KDD, 2018*
        </sub>](https://arxiv.org/abs/1803.02349)
        Code Expr
- **GCN**
        [<sub>
            *Semi-Supervised Classification with Graph Convolutional Networks, ICLR, 2017*
        </sub>](https://arxiv.org/abs/1609.02907)
        [Code](deep_recommenders/layers/gnn.py#L16) Expr
- **GraphSAGE**
        [<sub>
            *Inductive Representation Learning on Large Graphs, NIPS, 2017*
        </sub>](https://arxiv.org/abs/1706.02216)
        Code Expr
- **PinSage**
        [<sub>
            *Graph Convolutional Neural Networks for Web-Scale Recommender Systems, Pinterest, KDD, 2018*
        </sub>](https://arxiv.org/abs/1806.01973)
        Code Expr
- **IntentGC**
        [<sub>
            *IntentGC: a Scalable Graph Convolution Framework Fusing Heterogeneous Information for Recommendation, Alibaba, KDD, 2019*
        </sub>](https://arxiv.org/abs/1907.12377)
        Code Expr
- **GraphTR**
        [<sub>
            *Graph Neural Network for Tag Ranking in Tag-enhanced Video Recommendation, Tencent, CIKM, 2020*
        </sub>](https://dl.acm.org/doi/abs/10.1145/3340531.3416021)
        Code Expr
    
### NLP

- **Transformer**
        [<sub>
            *Attention Is All You Need, Google, NeurlPS, 2017*
        </sub>](https://arxiv.org/abs/1706.03762)
        [Code](deep_recommenders/layers/nlp/transformer.py) 
        [Expr](experiments/transformer.ipynb)



