

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
```