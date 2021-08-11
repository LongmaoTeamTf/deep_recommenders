# Deep Recommenders
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/6e9e50031edf46d4a1f4d39fa461b815)](https://app.codacy.com/gh/LongmaoTeamTf/deep_recommenders?utm_source=github.com&utm_medium=referral&utm_content=LongmaoTeamTf/deep_recommenders&utm_campaign=Badge_Grade_Settings)
[![Python](https://img.shields.io/badge/python-3.7-brightgreen)](requirements.txt)
[![TensorFlow](https://img.shields.io/badge/tensorflow-1.15_|_2.3-brightgreen)](requirements.txt)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Test](https://img.shields.io/badge/test-passing-brightgreen)]()

Deep Recommenders is an open-source recommendation system algorithm library 
built by `tf.estimator` and `tf.keras` that the advanced APIs of TensorFlow.
  
🤗️ This Library mainly used for self-learning and improvement, 
but also hope to help friends and classmates who are interested in the recommendation system to make progress together!

## Models

### Ranking

- **FM** 
        [[Code]](deep_recommenders/keras/layers/fm.py) 
        [<sub>
            *Factorization Machines, Osaka, 2010*
        </sub>](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
- **WDL**
        [<sub>
            *Wide & Deep Learning for Recommender Systems, Google, DLRS, 2016*
        </sub>](https://arxiv.org/abs/1606.07792)
- **PNN**
        [<sub>
            *Product-based Neural Networks for User Response Prediction, IEEE, 2016*
        </sub>](https://arxiv.org/abs/1611.00144)
- **FNN**
        [<sub>
            *Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction, RayCloud, ECIR, 2016*
        </sub>](https://arxiv.org/abs/1601.02376)
- **DeepFM** 
        [[Expr]](experiments/deepfm.ipynb) 
        [<sub>
            *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, Huawei, IJCAI, 2017*
        </sub>](https://www.ijcai.org/proceedings/2017/0239.pdf)
- **DCN** 
        [[Code]](deep_recommenders/keras/layers/dcn.py) 
        [<sub>
            *Deep & Cross Network for Ad Click Predictions, Google, KDD, 2017*
        </sub>](https://arxiv.org/abs/1708.05123)
- **xDeepFM** 
        [[Code]](deep_recommenders/keras/layers/xdeepfm.py) 
        [<sub>
            *xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, Microsoft, KDD, 2018*
        </sub>](https://arxiv.org/pdf/1803.05170.pdf)
- **DIN** 
        [[Code]](deep_recommenders/keras/layers/din.py) 
        [<sub>
            *Deep Interest Network for Click-Through Rate Prediction, Alibaba, KDD, 2018*
        </sub>](https://arxiv.org/abs/1706.06978)   
- **DIEN**
        [<sub>
            *Deep Interest Evolution Network for Click-Through Rate Prediction, Alibaba, AAAI, 2019*
        </sub>](https://arxiv.org/abs/1809.03672)
- **DLRM**
        [<sub>
            *Deep Learning Recommendation Model for Personalization and Recommendation Systems, Facebook, 2019*
        </sub>](https://arxiv.org/abs/1906.00091)

### Retrieval

- **DSSM**
        [<sub>
            *Learning Deep Structured Semantic Models for Web Search using Clickthrough Data, Microsoft, CIKM, 2013*
        </sub>](https://dl.acm.org/doi/10.1145/2505515.2505665)
- **YoutubeNet**
        [<sub>
            *Deep Neural Networks for YouTube Recommendations, Google, RecSys, 2016*
        </sub>](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)
- **SBC** 
        [[Code]](deep_recommenders/keras/layers/factorized_top_k.py) 
        [[Expr]](experiments/deep_retrieval.ipynb)
        [<sub>
            *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations, Google, RecSys, 2019*
        </sub>](https://dl.acm.org/doi/10.1145/3298689.3346996)
- **EBR**
        [<sub>
            *Embedding-based Retrieval in Facebook Search, Facebook, KDD, 2020*
        </sub>](https://arxiv.org/abs/2006.11632)
- **Item2Vec**
        [<sub>
            *Item2Vec: Neural Item Embedding for Collaborative Filtering, Microsoft, MLSP, 2016*
        </sub>](https://arxiv.org/vc/arxiv/papers/1603/1603.04259v2.pdf)
- **Airbnb**
        [<sub>
            *Real-time Personalization using Embeddings for Search Ranking at Airbnb, Airbnb, KDD, 2018*
        </sub>](https://dl.acm.org/doi/10.1145/3219819.3219885)
- **DeepWalk**
        [<sub>
            *DeepWalk: Online Learning of Social Representations, StonyBrook, KDD, 2014*
        </sub>](https://arxiv.org/abs/1403.6652)
- **EGES**
        [<sub>
            *Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba, Alibaba, KDD, 2018*
        </sub>](https://arxiv.org/abs/1803.02349)
- **GCN** 
        [[Code]](deep_recommenders/keras/layers/gnn.py#L16) 
        [[Expr]](experiments/gcn.ipynb)
        [<sub>
            *Semi-Supervised Classification with Graph Convolutional Networks, ICLR, 2017*
        </sub>](https://arxiv.org/abs/1609.02907)       
- **GraphSAGE**
        [<sub>
            *Inductive Representation Learning on Large Graphs, NIPS, 2017*
        </sub>](https://arxiv.org/abs/1706.02216)
- **PinSage**
        [<sub>
            *Graph Convolutional Neural Networks for Web-Scale Recommender Systems, Pinterest, KDD, 2018*
        </sub>](https://arxiv.org/abs/1806.01973)
- **IntentGC**
        [<sub>
            *IntentGC: a Scalable Graph Convolution Framework Fusing Heterogeneous Information for Recommendation, Alibaba, KDD, 2019*
        </sub>](https://arxiv.org/abs/1907.12377)
- **GraphTR**
        [<sub>
            *Graph Neural Network for Tag Ranking in Tag-enhanced Video Recommendation, Tencent, CIKM, 2020*
        </sub>](https://dl.acm.org/doi/abs/10.1145/3340531.3416021)
    
### Multi-task learning

- **MMoE**
        [<sub>
            *Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts, Google, KDD, 2018*
        </sub>](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)
- **ESMM**
        [<sub>
            *Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate, Alibaba, SIGIR, 2018*
        </sub>](https://arxiv.org/pdf/1804.07931.pdf)

### NLP

- **Word2Vec**
        [<sub>
            *Distributed Representations of Words and Phrases and their Compositionality, Google, NIPS, 2013*
        </sub>](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)

- **Transformer** 
        [[Code]](deep_recommenders/keras/layers/nlp/transformer.py) 
        [[Expr]](experiments/transformer.ipynb)
        [<sub>
            *Attention Is All You Need, Google, NeurlPS, 2017*
        </sub>](https://arxiv.org/abs/1706.03762)

- **BERT**
        [<sub>
            *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Google, NAACL, 2019*
        </sub>](https://arxiv.org/abs/1810.04805)
