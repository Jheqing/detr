"""
DETR 单张/多张图片推理脚本。
支持加载 training checkpoint 或官方预训练权重。
"""
import argparse
import random
import time
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from models import build_model
from util.box_ops import box_cxcywh_to_xyxy
from util.misc import NestedTensor

# 80 个 COCO 类别名称（与 COCO128 一致）
COCO_CLASSES_80 = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]

# YOLO 格式 80 类的顺序同样是从 0-79，名称同上
# 你也可以通过 --class_names 自定义

# ImageNet 归一化
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# 调色板
COLORS = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231", "#48F90A",
    "#92CC17", "#3DDB86", "#1A9334", "#00D4BB", "#2C99A8", "#00C2FF",
    "#344593", "#6473FF", "#0018EC", "#8438FF", "#520085", "#CB38FF",
    "#FF95C8", "#FF37C7",
]


def get_args_parser():
    parser = argparse.ArgumentParser('DETR Inference', add_help=False)
    parser.add_argument('--weights', required=True, type=str,
                        help='Path to checkpoint (.pth) file')
    parser.add_argument('--source', required=True, type=str,
                        help='Image path or directory of images')
    parser.add_argument('--output_dir', default='./runs/inference', type=str,
                        help='Output directory for visualization')
    parser.add_argument('--confidence', default=0.7, type=float,
                        help='Confidence threshold')
    parser.add_argument('--num_classes', default=80, type=int,
                        help='Number of classes')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--resize', default=800, type=int,
                        help='Input resize (short side)')
    parser.add_argument('--max_size', default=1333, type=int,
                        help='Maximum input size (long side)')

    # 如果需要覆盖类别名称
    parser.add_argument('--class_names', default=None, type=str, nargs='+',
                        help='Override class names (space-separated)')
    return parser


def load_model(args):
    """构建 DETR 模型并加载权重"""
    # 伪造一个 minimal args 给 build_model
    class MinimalArgs:
        pass

    m = MinimalArgs()
    m.dataset_file = 'yolo'
    m.num_classes = args.num_classes
    m.backbone = 'resnet50'
    m.dilation = False
    m.position_embedding = 'sine'
    m.enc_layers = 6
    m.dec_layers = 6
    m.dim_feedforward = 2048
    m.hidden_dim = 256
    m.dropout = 0.1
    m.nheads = 8
    m.num_queries = 100
    m.pre_norm = False
    m.masks = False
    m.aux_loss = True  # 推理不需要但 build 需要
    m.device = args.device
    m.set_cost_class = 1
    m.set_cost_bbox = 5
    m.set_cost_giou = 2
    m.bbox_loss_coef = 5
    m.giou_loss_coef = 2
    m.eos_coef = 0.1
    m.frozen_weights = None
    m.lr_backbone = 1e-5          # 推理时随便给个正值即可
    m.lr = 1e-4                   # build_model 可能用到

    model, _, postprocessors = build_model(m)
    model.to(args.device)

    # 加载 checkpoint
    ckpt = torch.load(args.weights, map_location='cpu')
    if 'model' in ckpt:
        state = ckpt['model']
    else:
        state = ckpt  # 纯权重文件

    # 可能 class_embed 维度不匹配，做自动适配
    weight_key = 'class_embed.weight'
    if weight_key in state:
        ckpt_nc = state[weight_key].shape[0] - 1
        tgt_nc = model.class_embed.out_features - 1
        if ckpt_nc != tgt_nc:
            print(f"[Inference] class_embed mismatch: ckpt={ckpt_nc}, target={tgt_nc}. Adapting...")
            new_w = torch.zeros(tgt_nc + 1, state[weight_key].shape[1])
            new_b = torch.zeros(tgt_nc + 1)
            n = min(ckpt_nc, tgt_nc)
            new_w[:n, :] = state[weight_key][:n, :]
            new_b[:n] = state['class_embed.bias'][:n]
            new_w[-1, :] = state[weight_key][-1, :]
            new_b[-1] = state['class_embed.bias'][-1]
            state[weight_key] = new_w
            state['class_embed.bias'] = new_b

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def resize_and_pad(image: Image.Image, target_size=800, max_size=1333):
    """等比例缩放 + 右下填充，返回 tensor 和 mask"""
    w, h = image.size
    # 等比例缩放
    scale = target_size / min(h, w)
    if max(scale * h, scale * w) > max_size:
        scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)

    # 转为 tensor 并归一化
    img_tensor = F.to_tensor(image)
    img_tensor = F.normalize(img_tensor, IMAGENET_MEAN, IMAGENET_STD)

    # 右下填充到 max_size（与训练时一致：resize 到 800 再让 NestedTensor 自动 pad）
    # NestedTensor 会自动创建 mask，这里我们也可以手动构造
    # 简单做法：不手动 pad，直接传 NestedTensor -> 但 DETR 的 forward 会自动处理
    # 我们这里直接传 tensor+mask
    c, hh, ww = img_tensor.shape
    mask = torch.zeros((hh, ww), dtype=torch.bool)
    return img_tensor.unsqueeze(0), mask.unsqueeze(0), (w, h)


@torch.no_grad()
def predict(model, image: Image.Image, device, target_size=800, max_size=1333):
    """对单张图片推理，返回预测结果列表及耗时统计"""
    timings = {}

    t0 = time.perf_counter()
    img_tensor, mask, orig_size = resize_and_pad(image, target_size, max_size)
    img_tensor = img_tensor.to(device)
    mask = mask.to(device)
    timings['preprocess'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    outputs = model(NestedTensor(img_tensor, mask))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    timings['inference'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    # 取最后一层输出
    pred_logits = outputs['pred_logits']       # [1, 100, num_classes+1]
    pred_boxes  = outputs['pred_boxes']         # [1, 100, 4]  cxcywh, 0~1

    prob = pred_logits.softmax(-1)              # [1, 100, num_classes+1]
    scores, labels = prob[..., :-1].max(-1)     # [1, 100], [1, 100]

    # 转换 boxes: cxcywh → xyxy，再缩放到原图尺寸
    boxes = box_cxcywh_to_xyxy(pred_boxes)      # [1, 100, 4] 0~1
    img_w, img_h = orig_size
    boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h], device=device)
    boxes = boxes.cpu()

    scores = scores.cpu()
    labels = labels.cpu()
    timings['postprocess'] = time.perf_counter() - t0

    return boxes[0], scores[0], labels[0], timings


def draw_results(image: Image.Image, boxes, scores, labels, class_names,
                 conf_thresh=0.7):
    """在图片上绘制检测框"""
    draw = ImageDraw.Draw(image)
    # 尝试加载字体，失败用默认
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for box, score, label in zip(boxes, scores, labels):
        if score < conf_thresh:
            continue
        x0, y0, x1, y1 = box.tolist()
        color = COLORS[int(label) % len(COLORS)]
        cls_name = class_names[int(label)] if int(label) < len(class_names) else f"cls_{int(label)}"

        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        text = f"{cls_name}: {score:.2f}"
        # 画文字背景
        text_bbox = draw.textbbox((x0, y0), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x0, y0), text, fill="white", font=font)

    return image


def main():
    parser = argparse.ArgumentParser('DETR Inference', parents=[get_args_parser()])
    args = parser.parse_args()

    class_names = args.class_names if args.class_names else COCO_CLASSES_80
    if len(class_names) < args.num_classes:
        for i in range(len(class_names), args.num_classes):
            class_names.append(f"class_{i}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    t0 = time.perf_counter()
    print(f"Loading model from: {args.weights}")
    model = load_model(args)
    model_load_time = time.perf_counter() - t0
    print(f"Model loaded. num_classes={args.num_classes}  [{model_load_time:.3f}s]")

    # 收集图片
    source = Path(args.source)
    if source.is_dir():
        img_paths = sorted(list(source.glob("*.jpg")) + list(source.glob("*.png")))
    else:
        img_paths = [source]

    if not img_paths:
        print("No images found!")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 累计耗时统计
    total_preprocess = 0.0
    total_inference  = 0.0
    total_postprocess = 0.0
    total_images = 0

    print(f"\n{'='*60}")
    for img_path in img_paths:
        print(f"\nProcessing: {img_path.name}")
        image = Image.open(img_path).convert("RGB")

        boxes, scores, labels, timings = predict(
            model, image, device,
            target_size=args.resize, max_size=args.max_size
        )

        # 过滤
        keep = scores >= args.confidence
        boxes  = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        total_preprocess  += timings['preprocess']
        total_inference   += timings['inference']
        total_postprocess += timings['postprocess']
        total_images += 1

        print(f"  ⏱  preprocess: {timings['preprocess']*1000:.1f}ms | "
              f"inference: {timings['inference']*1000:.1f}ms | "
              f"postprocess: {timings['postprocess']*1000:.1f}ms")
        print(f"  Detected {len(boxes)} objects:")
        for i in range(min(len(boxes), 20)):
            cls_name = class_names[int(labels[i])] if int(labels[i]) < len(class_names) else f"cls_{int(labels[i])}"
            box = boxes[i].tolist()
            print(f"    [{i}] {cls_name}: {scores[i]:.3f} @ ({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f})")

        result_img = draw_results(image, boxes, scores, labels, class_names,
                                  conf_thresh=args.confidence)
        save_path = output_dir / f"{img_path.stem}_det.jpg"
        result_img.save(save_path)
        print(f"  Saved to: {save_path}")

    # 汇总统计
    print(f"\n{'='*60}")
    print(f"[Summary] {total_images} image(s) processed")
    if total_images > 0:
        print(f"  Total model load time:   {model_load_time:.3f}s")
        print(f"  ── Per-image breakdown ──")
        print(f"    preprocess : {total_preprocess/total_images*1000:6.1f} ms/image")
        print(f"    inference  : {total_inference/total_images*1000:6.1f} ms/image")
        print(f"    postprocess: {total_postprocess/total_images*1000:6.1f} ms/image")
        total = (total_preprocess + total_inference + total_postprocess) / total_images
        print(f"    TOTAL      : {total*1000:6.1f} ms/image  ({1/total:.1f} FPS)")
    print(f"\nDone. Results saved to {output_dir}")


if __name__ == '__main__':
    main()
