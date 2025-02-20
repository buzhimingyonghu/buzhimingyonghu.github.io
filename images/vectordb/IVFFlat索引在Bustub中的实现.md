+++
date = '2024-11-30T19:42:02+08:00'
title = "IVFFlat索引在Bustub中的实现"
category = "技术"
tags = ["IVFFlat", "向量索引", "ANN"]
+++

### IVFFlat索引
IVF（Inverted File Index，倒排文件索引）是一种常用于向量搜索（ANN, Approximate Nearest Neighbor）的索引结构，主要用于高维数据检索，比如图像、文本、音频等向量数据的相似性搜索。

本文主要介绍IVFFlat索引在Bustub中的实现，主要从以下三个方面来回答：

1. IVFFlat索引的概述
2. IVFFlat索引在向量数据库中的应用
3. IVFFlat索引在Bustub中的实现


## IVFFlat索引的概述
IVF 通过聚类将向量划分到不同的“桶”（centroids, 聚类中心），查询时只在最相关的桶中搜索，从而减少计算量。

1. **训练阶段**：
   - 使用 **K-means** 聚类将所有数据点分成 \( K \) 个簇（每个簇有一个中心）。
   - 每个向量根据与哪个中心最近，归属于该中心对应的桶。

2. **索引构建**：
   - 记录每个桶中的向量 ID 及其原始向量。
   - 形成一个 **倒排表**，即每个簇对应多个向量。

3. **查询阶段**：
   - 先找到查询向量最接近的 \( N \) 个簇中心（通常 \( N \ll K \)）。
   - 只在这些簇对应的桶内进行精确搜索，而不是全量搜索。

---

### **示例**
假设有 100 万个 128 维向量：
- 用 K-means 训练出 **1000 个簇中心**（K=1000）。
- 每个向量归类到最近的簇，存入倒排表。

当查询时：
1. 计算查询向量与 1000 个簇中心的距离，找到最近的 10 个簇。
2. 只在这 10 个簇对应的桶里搜索最近邻，而不是在全部 100 万个向量中搜索。

这样，计算量大大减少，提高了搜索速度。

---

### **IVF 的优点**
✅ **高效查询**：比暴力搜索（Brute-force）快很多，适用于大规模数据。  
✅ **可扩展**：K 值可调，适应不同数据规模。  
✅ **支持 ANN（近似最近邻搜索）**：可以搭配其他方法（如 PQ, HNSW）进一步加速。

### **IVF 的缺点**
❌ **召回率下降**：仅搜索部分簇，可能会漏掉最优解。  
❌ **需要训练**：K-means 聚类需要预处理，适用于静态数据集。  


## IVF 在向量数据库中的应用
向量数据库的核心功能是**存储和检索高维向量**，而 IVF 作为索引结构，优化了搜索效率。以下是它的主要应用方式：

### 1. 数据索引
当你把向量数据插入数据库时：
1. **聚类（Clustering）**：
   - 使用 **K-means** 预训练 \( K \) 个聚类中心（centroids）。
   - 每个聚类中心代表一个“桶”（cell）。
   
2. **向量分桶（Assigning to Clusters）**：
   - 每个向量分配到最近的簇中心，存入该中心的倒排列表。

数据库内部结构示意：
```
Cluster 1: [vec_3, vec_7, vec_10]
Cluster 2: [vec_2, vec_6, vec_9]
...
Cluster K: [vec_1, vec_4, vec_8]
```

---

### 2. 向量查询
当你查询一个向量时：
1. **找到最接近的聚类中心**（一般选取 \( N \) 个最近的中心）。
2. **只在这些桶中搜索最近邻向量**（减少计算量）。
3. **返回最相似的向量（Top-k 结果）**。

举例：
- 查询向量 `q` 。
- 计算 `q` 到所有聚类中心的距离，找到最近的 3 个簇（假设 K=1000，N=3）。
- 只在这 3 个簇的倒排列表中搜索最近邻，而不是全局搜索。

---

## IVF 结合其他优化方法
IVF 可以和其他技术结合，进一步提升性能：
1. **IVF-PQ（Product Quantization）**：降低存储和计算成本。
2. **IVF-HNSW（Hierarchical Navigable Small World）**：加速近邻搜索。
3. **IVF-SQ（Scalar Quantization）**：减少索引占用的内存。





## IVFFlat索引实现技术

### 1. 类结构
```cpp
class IVFFlatIndex : public VectorIndex {
private:
    std::vector<Vector> centroids_;              // 聚类中心列表
    std::vector<std::vector<std::pair<Vector, RID>>> centroids_buckets_;  // 向量桶
    size_t lists_;                              // 聚类数量
    size_t probe_lists_;                        // 查询时检查的聚类数量
};
```

### 2. 索引构建流程
**实现步骤**：
1. 检查数据量是否足够
2. 初始化聚类桶
3. 随机选择初始聚类中心
4. 执行K-means迭代（最多500次）
5. 将向量分配到最近的聚类中心
```cpp
void IVFFlatIndex::BuildIndex(std::vector<std::pair<Vector, RID>> initial_data) {
  // 1. 检查数据量是否足够
  if (initial_data.size() < lists_) {
    return;
  }
  
  // 2. 初始化聚类桶
  centroids_buckets_.resize(lists_);
  
  // 3. 随机采样初始聚类中心
  centroids_ = RandomSample(initial_data, lists_);

  // 4. K-means迭代优化聚类中心
  for (size_t iter = 0; iter < max_iterations; ++iter) {
    centroids_ = FindCentroids(initial_data, centroids_, VectorExpressionType::L2Dist);
  }

  // 5. 将向量分配到最近的聚类中心
  for (const auto& pair : initial_data) {
    const Vector& vec = pair.first;
    size_t nearest_centroid_idx = FindCentroid(vec, centroids_, VectorExpressionType::L2Dist);
    centroids_buckets_[nearest_centroid_idx].push_back(pair);
  }
}
```

### 3. 插入流程
**实现步骤**：
1. 找到最近的聚类中心
2. 将向量和RID对插入对应的桶中
```cpp
void IVFFlatIndex::InsertVectorEntry(const std::vector<double> &key, RID rid) {
  // 1. 找到最近的聚类中心
  size_t nearest_centroid_idx = FindCentroid(key, centroids_, VectorExpressionType::L2Dist);
  
  // 2. 将向量添加到对应的聚类桶中
  centroids_buckets_[nearest_centroid_idx].emplace_back(key, rid);
}
```

### 4. 搜索流程
**实现步骤**：
1. 找到最近的probe_lists_个聚类中心
2. 在选中的聚类中搜索最近邻
3. 对结果排序
4. 返回前limit个最近邻的RID
```cpp
auto IVFFlatIndex::ScanVectorKey(const std::vector<double> &base_vector, size_t limit) {
  std::vector<RID> global_result;
  std::vector<std::pair<double, RID>> local_results;

  // 1. 找到最近的probe_lists_个聚类中心
  std::vector<size_t> nearest_centroids = FindNearestCentroids(base_vector, probe_lists_);

  // 2. 在选中的聚类中搜索最近邻
  for (size_t centroid_idx : nearest_centroids) {
    for (const auto& entry : centroids_buckets_[centroid_idx]) {
      const Vector& vec = entry.first;
      RID rid = entry.second;
      double distance = ComputeDistance(base_vector, vec, VectorExpressionType::L2Dist);
      local_results.emplace_back(distance, rid);
    }
  }

  // 3. 对候选结果排序
  std::sort(local_results.begin(), local_results.end());

  // 4. 返回top-k结果
  for (size_t i = 0; i < std::min(limit, local_results.size()); ++i) {
    global_result.push_back(local_results[i].second);
  }
  return global_result;
}
```

主要特点：

1. **数据组织**:
   - 使用K-means聚类将向量空间划分为多个区域
   - 每个区域有一个聚类中心(centroid)
   - 向量存储在最近的聚类中心对应的桶中

2. **优化策略**:
   - 查询时只需要搜索最近的几个聚类桶
   - 通过probe_lists_参数控制搜索范围
   - 在桶内进行精确距离计算

3. **关键参数**:
   - lists_: 聚类中心数量
   - probe_lists_: 查询时检查的聚类数量