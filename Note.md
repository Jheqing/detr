

# 修改

1、增加读取yolo格式数据功能

增加dataset中的yolo.py文件，大致内容是读取yolo格式转换成coco格式，然后用于模型训练。

修改main.py训练逻辑，使其能够兼容YOLO格式数据，增加yolo_cfg参数，获取数据集目录、类别数量和类别索引。

调整detr.py中构建模型的逻辑，增加适应数据集类别的动态构建方法。

2、修改评估使其兼容yolo格式评估

增加yolo.py里的get_yolo_coco_api，读取yolo数据，然后转换成coco，然后送给评估器进行评估。

在main.py里调整真实标签数据获取逻辑

```python
base_ds = get_yolo_coco_api(args.yolo_path, 'val.txt', getattr(args, 'class_names', None))

```



# 训练

使用yolo格式数据集进行训练
```python
python main.py --dataset_file yolo --yolo_cfg ./datasets/coco8-yolo.yaml --epoch 2 --batch_size 2 --num_workers 0 --lr 1e-4 --output_dir ./runs/train/exp1_coco8

python main.py --dataset_file yolo --yolo_cfg ./datasets/coco128.yaml --epoch 10 --batch_size 2 --num_workers 0 --lr 1e-4 --output_dir ./runs/train/exp2_coco128
```

## 使用预训练

修改代码使其能够根据num_classes的数量，在使用预训练权重的时候能够自适应的加载网络权重，注意如果num_classes超过预训练权重则保留原来权重的情况下添加随机初始化的部分，如果小于预训练权重，则相应的权重截取到num_classes。同时增加一个开关，可以在使用预训练权重的情况下主干网络等的冻结和解冻

```code
    构建空壳                            预训练权重(磁盘)
    ════════                           ════════════
    build_model(args)                  detr-r50-e632da11.pth
    ├── backbone                       ├── backbone.*       ← 形状固定(不依赖类别)
    ├── transformer                    ├── transformer.*    ← 形状固定(不依赖类别)
    ├── input_proj                     ├── input_proj.*     ← 形状固定
    ├── query_embed [100,256]          ├── query_embed.*    ← 形状固定
    ├── bbox_embed (MLP 256→256→4)     ├── bbox_embed.*     ← 形状固定
    └── class_embed [num_classes+1,256] └── class_embed.*   ← ⚠️ 唯一依赖类别的层!
                                                              COCO预训练=[92,256] (91类)
                                                              60类=[61,256]

```
main.py，新增以下功能：
两个新命令行参数（--freeze_backbone，--freeze_transformer）
自适应权重加载函数 adapt_pretrained_weights()
```code
    构建空壳                            预训练权重(磁盘)
    ════════                           ════════════
    build_model(args)                  detr-r50-e632da11.pth
    ├── backbone                       ├── backbone.*       ← 形状固定(不依赖类别)
    ├── transformer                    ├── transformer.*    ← 形状固定(不依赖类别)
    ├── input_proj                     ├── input_proj.*     ← 形状固定
    ├── query_embed [100,256]          ├── query_embed.*    ← 形状固定
    ├── bbox_embed (MLP 256→256→4)     ├── bbox_embed.*     ← 形状固定
    └── class_embed [num_classes+1,256] └── class_embed.*   ← ⚠️ 唯一依赖类别的层!
                                                              COCO预训练=[92,256] (91类)
                                                              60类=[61,256]

```
新的预训练权重加载流程
```code
                           args.resume 触发
                                │
                    ┌───────────┴───────────┐
                    │  Step 1: 加载权重文件   │
                    │  URL下载 或 本地读取     │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │  Step 2: 获取目标类别数 │
                    │  target_nc =           │
                    │  model.class_embed     │
                    │  .out_features - 1     │  ← 直接从已构建空壳取
                    └───────────┬───────────┘
                                │
                    ┌───────────┴─────────────────────────────────┐
                    │  Step 3: adapt_pretrained_class_embed()     │
                    │                                             │
                    │  预训练: class_embed.weight [92,256]        │
                    │          → pretrained_nc = 91               │
                    │  目标: class_embed.weight [61,256]          │
                    │          → target_nc = 60                   │
                    │                                             │
                    │  ┌──────────────────────────────────┐       │
                    │  │ 91 > 60  → 截断模式               │       │
                    │  │                                   │       │
                    │  │ 新 weight[0:60, :] = 旧[0:60, :]  │  ← 保留前60类
                    │  │ 新 weight[-1, :]   = 旧[-1, :]    │  ← 背景类复制
                    │  │ 旧 60~90 类 → 丢弃                │       │
                    │  └──────────────────────────────────┘       │
                    └───────────┬─────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │  Step 4: load_state_dict│
                    │  strict=False  → 安全   │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │  Step 5: 可选冻结       │
                    │  --freeze_backbone     │
                    │  --freeze_transformer   │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │  Step 6: 恢复训练状态   │
                    │  optimizer / scheduler  │
                    └───────────────────────┘

```

示例 1：60 类数据集，截断 COCO 预训练，只训练检测头
```python
python main.py --dataset_file yolo --yolo_cfg ./datasets/coco128.yaml --resum https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --freeze_backbone --freeze_transformer --epoch 10 --batch_size 2 --num_workers 0 --lr 1e-4 --output_dir ./runs/train/exp3_coco128_resume

python main.py --dataset_file yolo --yolo_cfg ./datasets/coco128.yaml --resum D://03_code//checkpoint//detr-r50-e632da11.pth --freeze_backbone --freeze_transformer --epoch 10 --batch_size 2 --num_workers 0 --lr 1e-4 --output_dir ./runs/train/exp3_coco128_resume
# 使用大学习率
python main.py --dataset_file yolo --yolo_cfg ./datasets/coco128.yaml --resum D://03_code//checkpoint//detr-r50-e632da11.pth --freeze_backbone --freeze_transformer --epoch 20 --batch_size 2 --num_workers 0 --lr 1e-2 --output_dir ./runs/train/exp3_coco128_resume

```
进一步微调

```python
python main.py --dataset_file yolo --yolo_cfg ./datasets/coco128.yaml --resum ./runs/train/exp3_coco128_resume/checkpoint.pth --freeze_backbone --start_epoch 0 --epoch 10 --batch_size 2 --num_workers 0 --lr 1e-2 --output_dir ./runs/train/exp4_coco128_resume_transformer


```


## 权重文件分析

```python
python analysispt.py --weights D://03_code//checkpoint//detr-r50-e632da11.pth
```


# 推理D:\03_code\temp
## 基本用法：单张图片推理
```python
python inference.py --weights ./runs/train/exp3_coco128_resume/checkpoint.pth --source D:/03_code/temp/000000000025.jpg --output_dir ./runs/inference
```
## 对整个目录推理

```python
python inference.py \
  --weights ./runs/train/exp4_coco128_resume_transformer/checkpoint.pth \
  --source D:/03_code/coco128-yolo/images/val2017/ \
  --confidence 0.5 --output_dir ./runs/inference
```

## 使用官方预训练权重 + CPU
```python
python inference.py \
  --weights D://03_code//checkpoint//detr-r50-e632da11.pth \
  --source ./test.jpg --num_classes 91 --device cpu
```

