+++
date = '2024-12-02T19:42:02+08:00'
title = "HNSW向量索引在Bustub中的实现"
category = "技术"
tags = ["HNSW", "向量索引", "ANN"]
+++


HNSW (Hierarchical Navigable Small World) 是一种高效的近似最近邻(ANN)搜索算法，特别适用于高维向量数据的相似度搜索，支持向量的插入和近邻搜索操作。
{{< figure src="/images/vectordb/1280X1280(1).PNG" title="HNSW结构示意图" >}}
{{< figure src="/images/vectordb/1280X1280.PNG" title="HNSW搜索过程" >}}

本文主要从以下两个方面介绍HNSW的实现：

1. 核心数据结构和算法实现
2. 向量检索和插入流程示例

## 核心数据结构和算法实现

### 1. 核心数据结构

#### 1.1 NSW (Navigable Small World)
```cpp
struct NSW {
  std::vector<Vector> &vertices_;         // 所有向量数据
  std::unordered_map<size_t, std::vector<size_t>> edges_;  // 邻接表
  std::vector<size_t> in_vertices_;       // 该层包含的顶点
  VectorExpressionType dist_fn_;          // 距离计算函数
  size_t m_max_;                         // 最大边数限制
}
```

#### 1.2 HNSWIndex
```cpp
class HNSWIndex {
  std::unique_ptr<std::vector<Vector>> vertices_;  // 向量数据
  std::vector<RID> rids_;                         // 记录ID
  std::vector<NSW> layers_;                       // 多层图结构
  
  // 配置参数
  size_t m_;                // 每个节点的邻居数
  size_t ef_construction_;  // 构建时的候选集大小
  size_t ef_search_;       // 搜索时的候选集大小
  double m_l_;             // 层级计算参数
}
```

### 2. 核心算法实现

#### 2.1 搜索算法
```cpp
auto NSW::SearchLayer(const vector<double> &query, size_t limit, const vector<size_t> &entry_points)
```
- 采用贪心搜索策略
- 使用优先队列维护候选集和结果集
- 通过距离比较进行剪枝优化
- 搜索过程：
  1. 从入口点开始搜索
  2. 遍历当前节点的邻居
  3. 更新候选集和结果集
  4. 当候选集最小距离大于结果集最大距离时终止

#### 2.2 插入算法
```cpp
void HNSWIndex::InsertVectorEntry(const vector<double> &key, RID rid)
```
- 插入步骤：
  1. 生成随机层级
  2. 从最高层开始搜索合适的插入位置
  3. 在目标层及以下建立连接
  4. 优化各层的邻居连接
  5. 必要时创建新层

### 3. 重要参数说明

- `m`：每个节点的最大邻居数
- `ef_construction`：构建索引时的候选集大小
- `ef_search`：查询时的候选集大小
- `m_max_`：非底层的最大边数
- `m_max_0_`：底层的最大边数（= m * m）

## 向量检索和插入流程示例

1. **索引结构**:
```
层级3:   o---o  (稀疏连接)
层级2:   o---o---o  (中等密度连接)
层级1:   o---o---o---o  (较密连接)
层级0:   o---o---o---o---o---o  (最密连接，包含所有节点)
```

2. **插入流程** (`InsertVectorEntry`):
```cpp
void HNSWIndex::InsertVectorEntry(const std::vector<double> &key, RID rid) {
  // 1. 随机决定新节点的最高层级
  int target_level = GenerateRandomLevel();
  
  // 2. 将向量数据和RID添加到存储中
  auto vertex_id = AddVertex(key, rid);
  
  // 3. 自顶向下插入过程
  if (!layers_[0].in_vertices_.empty()) {
    // 3.1 从最高层开始搜索
    std::vector<size_t> entry_points = {最高层的入口点};
    
    // 3.2 在高于目标层的层中只更新entry_points
    for (level = max_level; level > target_level; level--) {
      找到当前层最近的节点;
      将这些节点作为下一层的入口点;
    }
    
    // 3.3 在目标层及以下的层中建立连接
    for (; level >= 0; level--) {
      找到当前层最近的ef_construction个节点;
      选择最近的m个作为邻居;
      建立双向连接;
      优化邻居的连接(确保不超过最大连接数);
    }
  }
  
  // 4. 如果需要，创建新的层
  while (layers_.size() <= target_level) {
    创建新层并添加当前节点;
  }
}
```

3. **搜索流程** (`ScanVectorKey`):
```cpp
auto HNSWIndex::ScanVectorKey(const std::vector<double> &query, size_t k) {
  // 1. 从最高层开始
  entry_points = {最高层的入口点};
  
  // 2. 逐层向下搜索
  for (level = max_level; level > 0; level--) {
    // 在当前层找到最近的节点
    entry_points = layers_[level].SearchLayer(query, k, entry_points);
  }
  
  // 3. 在底层进行最终搜索
  final_results = layers_[0].SearchLayer(query, k, entry_points);
  
  // 4. 转换结果为RID并返回
  return 转换为RID列表;
}
```

4. **单层搜索流程** (`SearchLayer`):
```cpp
auto NSW::SearchLayer(const vector<double> &query, size_t k, const vector<size_t> &entry_points) {
  // 1. 初始化搜索状态
  candidate_queue = 空队列;
  result_set = 空优先队列;
  visited = 空集合;
  
  // 2. 将入口点加入候选集
  for (entry_point : entry_points) {
    计算距离;
    加入候选队列和结果集;
  }
  
  // 3. 贪心搜索
  while (!candidate_queue.empty()) {
    current = candidate_queue.front();
    
    // 3.1 获取当前节点的邻居
    neighbors = edges_[current];
    
    // 3.2 处理每个未访问的邻居
    for (neighbor : neighbors) {
      if (已访问) continue;
      
      计算距离;
      更新结果集;
      加入候选队列;
    }
    
    // 3.3 提前终止检查
    if (候选集中最近距离 > 结果集中最远距离) {
      break;
    }
  }
  
  // 4. 返回结果
  return 最近的k个节点;
}
```
