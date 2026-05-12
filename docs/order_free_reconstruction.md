# 无序重建 MVP V1

## 背景动机
当输入图像具有正确的时间顺序时，`Pi3 + VGGT-Long` 的效果很好。它的 chunk overlap 策略默认相邻 chunk 在时间上也相邻，因此它依赖时序邻接来维持几何一致性。一旦图像顺序被打乱，这种 overlap 就不再对应真实的几何邻接关系，重建质量会迅速下降。

这个 MVP 的目标**不是**恢复唯一正确的时间序列，而是直接从无序图像中构建一个适合后续重建的 chunk graph。

## 目标
实现一条无需训练的新 pipeline：

`descriptors -> view graph -> geometry-oriented clusters -> bridge discovery -> bridge-aware chunks`

核心原则：
- 图像相似度只用于候选召回
- 不使用 indoor/outdoor 语义切分
- bridge frames 会被多个 chunk 共享，以便后续 chunk 对齐更稳定
- 输出结果是一个 chunk graph，而不是恢复出的时间顺序

## Pipeline
### 1. Descriptor 提取
- 优先复用 `LoopModels` 中已有的 `SALAD + DINOv2` retrieval 路线
- 如果 SALAD 不可用，则退回到 DINO 全局 token
- 如果 `torch` 或权重都不可用，则再退回到轻量 OpenCV/PIL 描述子

### 2. View graph
- 基于全局 descriptor 构建 kNN graph
- 可选 mutual-kNN 过滤
- V1 只使用 appearance-only 的边权
- 为下一阶段的 geometry verification 预留接口

### 3. Geometry-oriented clusters
- 过滤低权重边
- 计算 connected components
- 对过大的 component 用 greedy region growing 拆分
- 尝试把过小 component 并入连接最强的邻居

### 4. Bridge discovery
- 对每一对相邻 cluster，计算候选 bridge frames
- bridge frame 可以来自 cluster A、cluster B，或同时连接到 A/B 的外部节点
- 对每个 cluster 对选出 top-M bridge frames

### 5. Bridge-aware chunks
- 每个 chunk 保留自己的 cluster core
- 相邻 chunk 共享 bridge frames
- 如果 chunk 太大，优先保留 bridge frames，再裁剪 core

## 运行命令
在 `VGGT-Long` 目录下运行：

```bash
python tools/order_free_reconstruct.py \
  --image_dir /path/to/images \
  --output_dir /path/to/output \
  --backbone pi3 \
  --max_chunk_size 80 \
  --min_chunk_size 20 \
  --knn 10 \
  --bridge_top_m 12 \
  --mutual_knn true \
  --align_mode graph
```

可选参数：

```bash
python tools/order_free_reconstruct.py \
  --image_dir /path/to/images \
  --output_dir /path/to/output \
  --use_geom_verification
```

`--use_geom_verification` 在 V1 中可以传入并记录到日志里，但暂时不会真正执行。

## 输出结果
输出目录结构如下：

```text
output_dir/
  chunks.json
  view_graph.json
  chunk_graph.json
  edge_scores.csv
  bridge_frames.json
  logs/
    run_config.json
    summary.json
    fallbacks.json
    cluster_stats.json
  reconstruction/
  visualizations/
```

重要文件说明：
- `view_graph.json`：图像节点、候选边、descriptor 提取器信息
- `edge_scores.csv`：每条边的 appearance 分数，以及预留的 geometry 分数字段
- `bridge_frames.json`：每对 cluster 之间选出的 bridge frames
- `chunks.json`：最终构建出的 bridge-aware chunks
- `chunk_graph.json`：基于共享图像建立的 chunk 邻接关系

## V1 当前限制
- 还没有接入 chunk 内部的 `Pi3` 或 `VGGT` 局部重建
- 还没有实现 chunk-to-chunk `Sim(3)` 对齐
- 还没有实现全局 chunk graph 同步
- `geom_score`、`uncertainty`、`sim3`、`residual` 目前只是 schema 占位

这些内容会在下一阶段接入，并复用 `Pi3Adapter` 与 `VGGT-Long` 现有的 `Sim(3)` 工具链。
