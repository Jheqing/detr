import os
import json
from PIL import Image

# 路径配置
ROOT = "D:/11_working/202505-animal_fences/coco8"
OUT = "./datasets/coco8_coco"
os.makedirs(f"{OUT}/train2017", exist_ok=True)
os.makedirs(f"{OUT}/val2017", exist_ok=True)
os.makedirs(f"{OUT}/annotations", exist_ok=True)

# COCO 80类（与YOLO类别ID严格对应）
COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog',
    'horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag',
    'tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
    'baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup',
    'fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot',
    'hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table',
    'toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

def yolo2coco(split):
    img_dir = f"{ROOT}/images/{split}"
    lbl_dir = f"{ROOT}/labels/{split}"
    out_img_dir = f"{OUT}/{split}2017"

    images = []
    annotations = []
    ann_id = 1

    for img_id, fname in enumerate(sorted(os.listdir(img_dir))):
        if not fname.endswith(('jpg','jpeg','png')):
            continue
        # 复制图片到DETR目录
        src_img = os.path.join(img_dir, fname)
        dst_img = os.path.join(out_img_dir, fname)
        Image.open(src_img).save(dst_img)
        w, h = Image.open(src_img).size

        # 写入images字段
        images.append({
            "id": img_id,
            "file_name": fname,
            "width": w,
            "height": h
        })

        # 读取YOLO txt
        txt_fname = os.path.splitext(fname)[0] + ".txt"
        txt_path = os.path.join(lbl_dir, txt_fname)
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, "r") as f:
            lines = f.read().splitlines()

        for line in lines:
            cls_id, xc, yc, bw, bh = map(float, line.split())
            cls_id = int(cls_id)
            # 核心转换：归一化 → 像素xywh
            x = (xc - bw/2) * w
            y = (yc - bh/2) * h
            bw *= w
            bh *= h
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": [x, y, bw, bh],
                "area": bw * bh,
                "iscrowd": 0
            })
            ann_id += 1

    # 写入JSON
    data = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for i, n in enumerate(COCO_CLASSES)]
    }
    with open(f"{OUT}/annotations/instances_{split}2017.json", "w") as f:
        json.dump(data, f, indent=2)

yolo2coco("train")
yolo2coco("val")
print("✅ 转换完成：", OUT)