# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
YOLO dataset loader adapted for DETR.
Loads images and labels based on train.txt/val.txt lists.
Label format assumed: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)
"""
from pathlib import Path
import os
import torch
import torch.utils.data
from PIL import Image
import yaml

import datasets.transforms as T


class YoloDetection(torch.utils.data.Dataset):
    def __init__(self, img_folder, list_file, transforms, return_masks=False, class_names=None):
        """
        Args:
            img_folder: Root directory containing images.
            list_file: Path to train.txt or val.txt containing relative image paths.
            transforms: Transforms to apply.
            return_masks: Whether to return segmentation masks (not supported in standard YOLO, usually False).
            class_names: List of class names for mapping IDs to names (optional).
        """
        super(YoloDetection, self).__init__()
        self.img_folder = Path(img_folder)
        self.list_file = Path(list_file)
        self._transforms = transforms
        self.return_masks = return_masks
        self.class_names = class_names
        
        # Read image paths from the list file
        with open(self.list_file, 'r') as f:
            self.img_paths = [line.strip() for line in f.readlines() if line.strip()]
        
        self.prepare = ConvertYoloToCocoFormat(return_masks)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_folder / self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        # Derive label path from image path
        # Assumption: label files are in the same directory as images, with .txt extension
        # Or in a parallel 'labels' directory structure. 
        # Here we assume same directory for simplicity, common in many YOLO setups.
        # If your structure is images/img.jpg -> labels/img.txt, modify this logic.
        """
        从图像路径推导标签路径  
        假设：标签文件与图像位于同一目录中，且扩展名为 .txt。  
        或者位于并行的"labels"目录结构中。  
        此处为简化起见，假设两个目录相同，这在许多YOLO配置中很常见。  
        如果您的目录结构是 images/img.jpg -> labels/img.txt，请修改此逻辑。
        """
        try:
            # 将路径中的 'images' 替换为 'labels'
            label_relative_path = Path(str(self.img_paths[idx]).replace('images', 'labels'))
            label_path = self.img_folder / label_relative_path.with_suffix('.txt')
        except Exception:
            # 如果替换失败或不存在，回退到同目录查找        
            label_path = img_path.with_suffix('.txt')
        
        # Read labels
        target = {"boxes": [], "labels": []}
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_c, y_c, w, h = map(float, parts)
                        target["boxes"].append([x_c, y_c, w, h])
                        target["labels"].append(int(class_id))
        
        # Convert lists to tensors if not empty
        if target["boxes"]:
            target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
            target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            
        # Add dummy image_id based on index
        target["image_id"] = torch.tensor([idx])
        
        # Apply conversion and transforms
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
            
        return img, target


class ConvertYoloToCocoFormat(object):
    """
    Converts YOLO normalized center-format boxes to absolute corner-format boxes expected by DETR.
    YOLO format: cx, cy, w, h (normalized 0-1)
    DETR format: xmin, ymin, xmax, ymax (absolute pixels)
    将YOLO的归一化中心格式框转换为DETR所期望的绝对角格式框。  
    YOLO格式：cx, cy, w, h（归一化范围0-1）  
    DETR格式：xmin, ymin, xmax, ymax（绝对像素）
    """
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size
        
        boxes = target["boxes"]
        if len(boxes) > 0:
            # YOLO boxes are normalized: cx, cy, w, h
            # Convert to absolute pixels: x_center, y_center, width, height
            boxes[:, 0] *= w  # cx
            boxes[:, 1] *= h  # cy
            boxes[:, 2] *= w  # w
            boxes[:, 3] *= h  # h
            
            # Convert center-size to corner format: xmin, ymin, xmax, ymax
            # xmin = cx - w/2, ymin = cy - h/2
            # xmax = cx + w/2, ymax = cy + h/2
            boxes_new = torch.zeros_like(boxes)
            boxes_new[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # xmin
            boxes_new[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # ymin
            boxes_new[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # xmax
            boxes_new[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # ymax
            
            # Clamp boxes to image boundaries
            boxes_new[:, 0::2].clamp_(min=0, max=w)
            boxes_new[:, 1::2].clamp_(min=0, max=h)
            
            # Filter out invalid boxes (where area <= 0)
            keep = (boxes_new[:, 3] > boxes_new[:, 1]) & (boxes_new[:, 2] > boxes_new[:, 0])
            boxes = boxes_new[keep]
            labels = target["labels"][keep]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target["boxes"] = boxes
        target["labels"] = labels
        
        # DETR expects these fields
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        
        # Dummy fields for compatibility with COCO API evaluation if needed
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        
        if self.return_masks:
            # YOLO typically doesn't have masks, return empty
            target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)

        return image, target


def make_yolo_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    """
    Builds the YOLO dataset.
    Expects args.yolo_path to be set (usually derived from args.yolo_cfg in main.py).
    The path should contain:
    - train.txt
    - val.txt
    - images/ folder (or similar structure referenced in txt files)
    - data.yaml (optional, for class names)
    """
    root = Path(args.yolo_path)
    assert root.exists(), f'provided YOLO path {root} does not exist'
    
    # Assuming structure:
    # root/
    #   train.txt
    #   val.txt
    #   images/（在txt文件中引用的路径为相对路径，例如：images/train/001.jpg）或者txt文件中的路径是从根目录开始的完整相对路径
    #   data.yaml (optional)
    """
    # root/
    #   train.txt
    #   val.txt
    #   images/（在txt文件中引用的路径为相对路径，例如：images/train/001.jpg）或者txt文件中的路径是从根目录开始的完整相对路径
    """
    
    list_file = root / f'{image_set}.txt'
    assert list_file.exists(), f'list file {list_file} does not exist'
    
    # The img_folder is the root, because paths in txt are relative to root? 
    # Or if txt contains paths like "images/train/xxx.jpg", then img_folder should be root.
    # Let's assume img_folder is the root directory provided in args.yolo_path
    img_folder = root
    
    # 获取类别名称列表
    # 优先尝试从 data.yaml 加载
    class_names = getattr(args, 'class_names', None)
    yaml_path = root / 'data.yaml'
    if yaml_path.exists():
        try:
            with open(yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            if 'names' in data_config:
                names_dict = data_config['names']
                # Handle both list and dict formats for names in YOLO yaml
                if isinstance(names_dict, list):
                    class_names = names_dict
                elif isinstance(names_dict, dict):
                    # Sort by key to ensure order
                    class_names = [names_dict[k] for k in sorted(names_dict.keys())]
        except Exception as e:
            print(f"Warning: Could not load class names from {yaml_path}: {e}")

    dataset = YoloDetection(
        img_folder=img_folder, 
        list_file=list_file, 
        transforms=make_yolo_transforms(image_set), 
        return_masks=args.masks,
        class_names=class_names
    )
    return dataset


def get_yolo_coco_api(root_path, list_file_name, class_names=None):
    """
    Constructs a COCO-like API object from YOLO format data for evaluation.
    This mimics the structure expected by pycocotools.
    
    Args:
        root_path: Root directory of the YOLO dataset.
        list_file_name: Name of the list file (e.g., 'val.txt').
        class_names: List of class names. If None, tries to load from data.yaml.
        
    Returns:
        A fake COCO API object or a dictionary structure that can be used by evaluator.
        Note: The standard DETR evaluator expects a 'coco' object from pycocotools.coco.COCO.
        We will generate a temporary JSON structure and load it.
    """
    from pycocotools.coco import COCO
    import json
    import tempfile
    
    root = Path(root_path)
    list_file = root / list_file_name
    
    # Load class names if not provided
    if class_names is None:
        yaml_path = root / 'data.yaml'
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            if 'names' in data_config:
                names_dict = data_config['names']
                if isinstance(names_dict, list):
                    class_names = names_dict
                elif isinstance(names_dict, dict):
                    class_names = [names_dict[k] for k in sorted(names_dict.keys())]
    
    if class_names is None:
        raise ValueError("Class names must be provided or available in data.yaml")

    # Build COCO JSON structure
    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add categories
    for idx, name in enumerate(class_names):
        coco_dict["categories"].append({
            "id": idx,
            "name": name,
            "supercategory": "none"
        })
        
    # Read image list
    with open(list_file, 'r', encoding='utf-8') as f:
        img_paths = [line.strip() for line in f.readlines() if line.strip()]
        
    ann_id = 0
    img_id_map = {} # Map from index to COCO image ID
    for img_idx, rel_img_path in enumerate(img_paths):
        img_full_path = root / rel_img_path
        if not img_full_path.exists():
            print(f"Warning: Image not found: {img_full_path}")
            continue
            
        try:
            with Image.open(img_full_path) as img:
                w, h = img.size
        except Exception as e:
            print(f"Warning: Could not read image {img_full_path}: {e}")
            continue
            
        # Use img_idx as image ID to ensure consistency with dataset __getitem__
        # In YoloDetection.__getitem__, target["image_id"] = torch.tensor([idx])
        # So we must use the same index here.
        coco_image_id = img_idx
        
        # Add image info
        coco_dict["images"].append({
            "id": coco_image_id,
            "file_name": rel_img_path,
            "width": w,
            "height": h
        })
        img_id_map[img_idx] = True
        
        # Derive label path
        try:
            label_relative_path = Path(str(rel_img_path).replace('images', 'labels'))
            label_path = root / label_relative_path.with_suffix('.txt')
        except Exception:
            label_path = img_full_path.with_suffix('.txt')

        print(f"Processing image {img_full_path}, looking for label {label_path}")
        if label_path.exists():
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_c, y_c, bw, bh = map(float, parts)
                        
                        # Convert YOLO to COCO bbox format: x_min, y_min, width, height (absolute)
                        x_min = (x_c - bw / 2) * w
                        y_min = (y_c - bh / 2) * h
                        width = bw * w
                        height = bh * h
                        
                        # Clamp
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        width = min(width, w - x_min)
                        height = min(height, h - y_min)
                        
                        if width <= 0 or height <= 0:
                            continue
                            
                        coco_dict["annotations"].append({
                            "id": ann_id,
                            "image_id": coco_image_id,
                            "category_id": int(class_id),
                            "bbox": [x_min, y_min, width, height],
                            "area": width * height,
                            "iscrowd": 0
                        })
                        ann_id += 1
                        
    print(f"YOLO COCO API: Loaded {len(coco_dict['images'])} images and {len(coco_dict['annotations'])} annotations.")
    
    if len(coco_dict['images']) == 0:
        raise ValueError("No valid images found in the validation set. Check your val.txt and image paths.")

    # Create temporary JSON file and load into COCO API
    # Using a temp file is robust for the COCO API loader
    tmp_json = tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False)
    json.dump(coco_dict, tmp_json)
    tmp_json.close()
    
    coco_api = COCO(tmp_json.name)
    
    # Clean up temp file optionally, but COCO might keep reference. 
    # Usually safe to delete after loading if loaded into memory, but COCO pyload might lazy load.
    # To be safe, we can leave it or delete. Let's try to delete.
    import os
    try:
        os.unlink(tmp_json.name)
    except:
        pass
        
    return coco_api
