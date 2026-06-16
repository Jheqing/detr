
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












# 测试



# 推理
