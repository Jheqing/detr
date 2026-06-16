
# 数据集转换
coco8_yolo转换成coco8_coco的json格式。
原理coco8_yolo文件目录如下
```
./datasets/coco8
├── images/
│   ├── train/  # 4张jpg
│   └── val/    # 4张jpg
└── labels/
    ├── train/  # 4个txt（YOLO归一化）
    └── val/

```
使用yolo2coco.py将yolo格式转换成coco格式
coco8_coco的json格式如下，注意修改类别。

```
datasets/coco8_coco
├── images/
│   ├── train2017/  # 4张
│   └── val2017/    # 4张
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

# 修改路径

# 调整detr.py中的num_classes
在models的detr.py中修改num_classes

```python
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    num_classes = 20 if args.dataset_file != 'coco' else 80
```

# 训练

python命令训练coco8_coco数据集
```python
python main.py --dataset_file coco --coco_path ./datasets/coco8_coco/ --epoch 2 --batch_size 2 --num_workers 0 --lr 1e-4 --output_dir ./runs/train/exp1_coco8
```

# 日志和结果分析

- coco8_coco是一个小数据集（只有8张图：4张训练，4张验证）
- 训练配置：2个epoch，batch_size=2，lr=0.0001



1. **基本配置信息**：
   - 非分布式模式
   - lr=0.0001, lr_backbone=1e-5
   - batch_size=2
   - epochs=2 (只训练了2个epoch)
   - backbone=resnet50
   - num_queries=100
   - 数据集路径：./datasets/coco8_coco/
   - 输出目录：./runs/train/exp1_coco8

2. **模型参数**：41,299,541 参数量（约41M）

3. **训练过程分析**：

**Epoch 0:**

- [0/2]: loss=61.89, class_error=100%, loss_ce=3.92, loss_bbox=4.15, loss_giou=1.86
- [1/2]: loss=51.93(平均56.91), class_error=100%, loss_ce=2.37(平均3.15)
- 平均loss: 51.93(平均56.91)

**Epoch 0 Test:**

- AP全部为0.000
- loss=44.66

**Epoch 1:**

- [0/2]: loss=48.32, class_error=100%
- [1/2]: loss=44.83(平均46.57), class_error=100%
- 平均loss: 44.83(平均46.57)

**Epoch 1 Test:**

- AP仍然全部为0.000
- loss=44.13

关键问题分析：

1. **class_error始终是100%** - 分类完全错误
2. **AP全是0** - 没有检测到任何目标
3. **只有2个epoch的训练** - DETR通常需要300+ epochs才能收敛
4. **数据集太小** - 只有4张训练图片



```text
┌─────────────────────────────────────────────────────────────┐
│  硬件/环境                                                   │
│  ├── 设备: CUDA (GPU)                                       │
│  ├── 显存峰值: 2373 MB                                      │
│  └── 分布式: 单卡模式 (world_size=1)                         │
├─────────────────────────────────────────────────────────────┤
│  模型配置                                                    │
│  ├── Backbone: ResNet50 (ImageNet预训练)                    │
│  ├── Encoder: 6层, Decoder: 6层                             │
│  ├── Hidden dim: 256, FFN: 2048, Heads: 8                  │
│  ├── Object Queries: 100                                    │
│  ├── 总参数量: 41,299,541 (~41M)                            │
│  └── 辅助损失: 开启 (aux_loss=True, 5层decoder输出)          │
├─────────────────────────────────────────────────────────────┤
│  训练超参                                                   │
│  ├── Epochs: 2 (极少量！)                                   │
│  ├── Batch Size: 2                                          │
│  ├── LR: 1e-4 (Transformer) / 1e-5 (Backbone)              │
│  ├── Weight Decay: 1e-4                                     │
│  ├── Clip Grad Norm: 0.1                                    │
│  └── 数据集: coco8_coco (4 train + 4 val = 仅8张图!)        │
└─────────────────────────────────────────────────────────────┘
```



```
Epoch: [0]  [0/2]  eta: 0:00:26  lr: 0.000100  class_error: 100.00  loss: 61.8871 (61.8871)  loss_ce: 3.9226 (3.9226)  l: 4.5779 (4.5779)  loss_bbox_0: 4.2143 (4.2143)  loss_giou_0: 1.8471 (1.8471)  loss_ce_1: 4.3007 (4.3007)  loss_bbox_1:  (4.3705)  loss_bbox_2: 4.1904 (4.1904)  loss_giou_2: 1.8487 (1.8487)  loss_ce_3: 4.4080 (4.4080)  loss_bbox_3: 4.1781 ()  loss_bbox_4: 4.2617 (4.2617)  loss_giou_4: 1.8265 (1.8265)  loss_ce_unscaled: 3.9226 (3.9226)  class_error_unscaled: iou_unscaled: 0.9312 (0.9312)  cardinality_error_unscaled: 93.5000 (93.5000)  loss_ce_0_unscaled: 4.5779 (4.5779)  loss_0.9236)  cardinality_error_0_unscaled: 95.0000 (95.0000)  loss_ce_1_unscaled: 4.3007 (4.3007)  loss_bbox_1_unscaled: 0.8_error_1_unscaled: 95.0000 (95.0000)  loss_ce_2_unscaled: 4.3705 (4.3705)  loss_bbox_2_unscaled: 0.8381 (0.8381)  loss_g5.0000 (95.0000)  loss_ce_3_unscaled: 4.4080 (4.4080)  loss_bbox_3_unscaled: 0.8356 (0.8356)  loss_giou_3_unscaled: 0.91ss_ce_4_unscaled: 4.0116 (4.0116)  loss_bbox_4_unscaled: 0.8523 (0.8523)  loss_giou_4_unscaled: 0.9132 (0.9132)  cardina.1572  max mem: 1111
```

## 耗时分析

 **Epoch 0（第1个epoch）的训练耗时统计**：

```text
Epoch: [0] Total time: 0:00:14 (7.0083 s / it)
 │        │       │         │          │
 │        │       │         │          └── 每次迭代平均 7.0083 秒
 │        │       │         └── 每次迭代 (iteration)
 │        │       └── 总耗时 14 秒
 │        └── 第 0 个 epoch (从0开始计数)
 └── 训练轮次
```

| 指标           | 值                     | 含义                              |
| -------------- | ---------------------- | --------------------------------- |
| **Total time** | `0:00:14` = **14秒**   | 跑完整个 Epoch 0 的墙钟时间       |
| **s / it**     | **7.0083 秒/步**       | 平均每次迭代的耗时                |
| **总迭代次数** | 14 ÷ 7.0083 ≈ **2 次** | = 数据集图片数(4) ÷ batch_size(2) |

从之前的训练日志可以看到：

```
Epoch: [0]  [0/2]   ← 第1个iteration (step 0，共2步)
Epoch: [0]  [1/2]   ← 第2个iteration (step 1)
Epoch: [0] Total time: 0:00:14 (7.0083 s / it)  ← 结束
```

- 你的数据集：**4 张训练图**
- `batch_size=2`
- 所以每个 Epoch 只有 **4 ÷ 2 = 2 个 iteration**

这行表示第 0 个 epoch 总共花了 **14 秒**跑完 2 个 batch，平均每步 **7 秒**。其中第 1 步因 CUDA 预热特别慢（13.4s），第 2 步就快多了（0.4s），到 Epoch 1 更是稳定在 **0.4 秒/步**。



Averaged stats ：平均数据

Accumulating evaluation results...：正在累积评估结果……

IoU metric: bbox：IoU指标，边界框

工程流程已跑通✅，当前结果符合 DE TR 冷启动预期。要看到非零 AP，建议至少训练 **50+ epoch** 或直接从 **COCO 预训练权重**微调。

- 用 `--resume=detr_resnet50_dc5.pth` 从预训练开始微调
- loss 降低 ≠ 检测准确，需等 confidence 超过阈值

## 2个警告

### 警告 1：`pretrained` 参数已弃用

```text
File "torchvision\models\_utils.py", line 208
UserWarning: The parameter 'pretrained' is deprecated since 0.13, 
please use 'weights' instead.
```

DETR 的 backbone 加载代码（`backbone.py`）使用了**旧版 API**：
Now I can see the exact code causing these warnings. Line 89-91 in backbone.py uses `pretrained=is_main_process()` which is the old API. Let me explain both warnings clearly.
找到了！问题就在 `backbone.py:89-91`：

```python
# 第 89-91 行 — 使用了旧版 torchvision API
backbone = getattr(torchvision.models, name)(
    replace_stride_with_dilation=[False, False, dilation],
    pretrained=is_main_process(),   # ⚠️ 旧参数名
    norm_layer=FrozenBatchNorm2d)
```

#### 警告 1：`pretrained` → `weights`

| 版本             | API 写法                           |
| ---------------- | ---------------------------------- |
| **旧版 (≤0.12)** | `pretrained=True`                  |
| **新版 (≥0.13)** | `weights=ResNet50_Weights.DEFAULT` |

```python
# DETR 当前写法（旧版，会触发警告）:
torchvision.models.resnet50(pretrained=True)

# 新版推荐写法:
torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)

# 或用最新默认权重:
torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
```

### 警告 2：字符串参数被弃用

```text
Arguments other than a weight enum or `None` for `weights` are deprecated
current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`
```

这是说：`weights` 参数现在应该传**枚举类型**（Enum）而不是字符串或布尔值。

| 项目         | 说明                                                   |
| ------------ | ------------------------------------------------------ |
| **严重程度** | 🟢 **纯 Warning，不影响任何功能**                       |
| **功能影响** | 无。代码行为完全一致，只是写法过时                     |
| **输出污染** | 每次启动会打印两行警告信息，不干扰训练结果             |
| **何时移除** | torchvision 未来版本可能会彻底移除旧参数（目前仍兼容） |

如果想消除警告，修改 `models/backbone.py:89-91`：

```python
# === 修改前（旧版 API）===
backbone = getattr(torchvision.models, name)(
    replace_stride_with_dilation=[False, False, dilation],
    pretrained=is_main_process(),
    norm_layer=FrozenBatchNorm2d)

# === 修改后（新版 API）===
from torchvision.models import ResNet50_Weights, ResNet101_Weights

_weights = ResNet50_Weights.IMAGENET1K_V1 if is_main_process() else None
if name == 'resnet101':
    _weights = ResNet101_Weights.IMAGENET1K_V1 if is_main_process() else None

backbone = getattr(torchvision.models, name)(
    replace_stride_with_dilation=[False, False, dilation],
    weights=_weights,
    norm_layer=FrozenBatchNorm2d)
```

> **建议**：这两个 warning 可以直接忽略。DETR 原版代码发布时 torchvision 还是旧版 API，属于**历史遗留问题**，Facebook 官方仓库至今未更新这行代码。训练结果完全不受影响。

## `number of params: 41,299,541` 解释

这行表示 DETR 模型的**总参数量（可训练权重个数）**：**约 4130 万 = 41.3M**，参数分布拆解

```
DETR 总参数: 41,299,541 (~41.3M)
│
├── 🏗️ ResNet50 Backbone:  ~25.6M (62%)
│   ├── 卷积层权重 (Conv2d)
│   ├── FrozenBatchNorm (冻结的BN统计量，不参与梯度)
│   └── layer1~layer4 特征提取
│
├── 🔄 Transformer:          ~11.8M (29%)
│   ├── Encoder (6层):  自注意力 + FFN
│   ├── Decoder (6层):  交叉自注意力 + FFN
│   └── 每层的 Q/K/V 投影矩阵
│
└── 🎯 预测头:              ~3.9M  (9%)
    ├── class_embed:    256 → 92 (分类头，92个COCO类别+背景)
    ├── bbox_embed:     MLP 256→256→4 (回归头，预测bbox坐标)
    └── query_embed:    100×256 (可学习的Object Query嵌入)
```

直观对比

| 模型                | 参数量    | 大小(fp32)  |
| ------------------- | --------- | ----------- |
| **DETR (ResNet50)** | **41.3M** | **~157 MB** |
| YOLOv8n             | 3.2M      | ~12 MB      |
| YOLOv8x             | 68.2M     | ~260 MB     |
| ResNet-50 (纯)      | 25.6M     | ~98 MB      |
| ViT-Base            | 86M       | ~328 MB     |

为什么是 41299541 这个精确数字？这行输出来自 `main.py` 中模型构建后的参数计数：

```python
# main.py 中类似这样的代码:
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"number of params: {n_parameters}")
# → 41299541
```

每个参数都是一个浮点数（float32 = 4字节），所以：

- **模型文件大小** ≈ 41,299,541 × 4 bytes **≈ 157 MB**（保存为 `.pth` 时还会压缩）

> **一句话总结**：这就是整个 DETR 网络（ResNet50骨干 + Transformer编解码器 + 检测头）中所有可学习权重的总数，约 **4130万个参数 / 占显存约 157MB（fp32）**。



## 预训练权重分析

你的训练**只用了 Backbone 的 ImageNet 预训练**，没有用完整的 **DETR COCO 预训练权重**。使用了`resnet50-0676ba61.pth`预训练权重，位置在`C:\Users\Administrator/.cache\torch\hub\checkpoints\resnet50-0676ba61.pth`。未使用：DETR COCO 全量预训练

```
Namespace(
  resume='',           # ← 空 = 没有恢复检查点
  frozen_weights=None, # ← 空 = 没有冻结权重
  start_epoch=0        # ← 从 0 开始 = 从头训练
)
```

这意味着 **Transformer + 检测头完全从随机初始化开始学习**！

```
┌─────────────────────────────────────────────────────────────────┐
│                    预训练权重的三个层次                          │
├─────────────┬───────────────────┬──────────────────────────────┤
│  层次 1     │  层次 2            │  层次 3                      │
│  (你当前)   │                   │                              │
├─────────────┼───────────────────┼──────────────────────────────┤
│  ResNet50   │  完整 DETR         │  冻结 DETR                  │
│  ImageNet   │  COCO 预训练       │  (仅训分割头)                │
│  预训练     │                   │                              │
│             │                   │                              │
│  仅 Backbone│  Backbone+        │  冻结上述全部                 │
│  有预训练   │  Transformer+     │  只训练 segmentation head    │
│             │  检测头            │                              │
│             │  都有预训练        │                              │
├─────────────┼───────────────────┼──────────────────────────────┤
│  ~25.6M 参数│  ~41.3M 全部      │  ~41.3M 冻结 + 新增分割头    │
│  文件: 98MB │  文件: ~160MB     │  文件: ~160MB                │
│  来源: torchvision │ 来源: Facebook | 来源: 自己训练的检测模型  │
└─────────────┴───────────────────┴──────────────────────────────┘

```

## 如何使用完整 DETR COCO 预训练权重

### 通过 PyTorch Hub（推荐）

```python
import torch

# 一行代码加载完整预训练 DETR
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)

```

这个应该需要修改代码。

## 通过 `--resume` 参数（最简单）

直接在命令行指定 Facebook 官方的预训练权重 URL 或本地路径：

```
python main.py \
  --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
  --coco_path ./datasets/coco8_coco/ \
  --epochs 10 \
  # ... 其他参数

```

**支持的预训练模型**（来自 `hubconf.py`）：

| 模型命令                     | AP       | AP50     | 权重文件                  |
| ---------------------------- | -------- | -------- | ------------------------- |
| `detr-r50-e632da11.pth`      | **42.0** | 62.4     | ResNet50, 标准 DETR ⭐推荐 |
| `detr-r50-dc5-f0fb7ef5.pth`  | 43.3     | 63.1     | ResNet50 + DC5 空洞卷积   |
| `detr-r101-2c7b67e5.pth`     | 43.5     | 63.8     | ResNet101                 |
| `detr-r101-dc5-a2e86def.pth` | **44.9** | **64.7** | ResNet101 + DC5 (最强)    |

### 方法3 先下载权重到本地

```bash
# 手动下载（约160MB）
curl -o detr-r50.pth https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth

# 然后使用本地路径
python main.py --resume detr-r50.pth --epochs 10 ...
```

### 注意事项：num_classes 不匹配问题

⚠️ **关键提醒**：Facebook 预训练模型的 `num_classes=91`（COCO 80类 + 背景）。如果你的数据集类别数不同，有两种处理方式：

```python
# 方式 A：保持 91 类，只替换最后分类层（推荐）
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)

# 替换分类头为你的类别数（假设你有 N 类 + 1 背景）
model.class_embed = torch.nn.Linear(256, N + 1)

# 冻结其他部分，只训练新分类头
for param in model.parameters():
    param.requires_grad = False
for param in model.class_embed.parameters():
    param.requires_grad = True
```

> **建议你的下一步操作**：
>
> ```bash
> python main.py --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
> --coco_path ./datasets/coco8_coco/ --epochs 10
> ```
>
> 这样可以直接在 coco8 数据集上看到**非零 AP**，验证整个流程是否正常！















# 测试





# 推理
