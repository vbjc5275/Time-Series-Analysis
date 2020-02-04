# 時間序列的相似度衡量

## 前言
在時間序列的研究中，諸多數據挖掘任務如分類、分群等，皆仰賴時間序列之間相似性的度量，如何計算序列之間存在的差異大小，是非常重要且基礎問題。
## 種類
相似性度量在於分類上主要分爲距離度量和相似度度量。
### 距離度量
距離度量（Distance）用於衡量個體在空間上存在的距離。
1. 歐幾里得距離(Euclidean Distance)
2. 明可夫斯基距離(Minkowski Distance)
3. 曼哈頓距離(Manhattan Distance)
4. 切比雪夫距離(Chebyshev Distance)
5. 馬哈拉諾比斯距離(Mahalanobis Distance)
### 相似度度量
相似度度量（Similarity），即計算個體間的相似程度，與距離度量相反，相似度度量的值越小，說明個體間相似度越小，差異越大。
1. 向量空間餘弦相似度(Cosine Similarity)
2. 皮爾森相關係數(Pearson Correlation Coefficient)
3. Jaccard相似係數(Jaccard Coefficient)
4. 調整餘弦相似度(Adjusted Cosine Similarity)

