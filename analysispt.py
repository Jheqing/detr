#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.pth / .pt 权重文件分析工具
支持：
  - 可视化输出权重文件的整体结构
  - 各模块详细信息（层名、shape、参数量、内存占用）
  - 检测 checkpoint 类型（纯模型/完整训练状态）
  - 与当前模型对比差异
  - 统计汇总

用法：
    # 基本分析某个权重文件
    python analysispt.py --weights path/to/checkpoint.pth

    # 详细模式（列出所有 tensor）
    python analysispt.py --weights path/to/checkpoint.pth --detail

    # 与现有模型对比
    python analysispt.py --weights path/to/checkpoint.pth --compare

    # 指定颜色输出（默认自动检测终端）
    python analysispt.py --weights path/to/checkpoint.pth --no-color
"""

import argparse
import collections
import os
import sys
import math
from collections import defaultdict

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════
# 终端颜色（跨平台）
# ═══════════════════════════════════════════════════════════════
class Colors:
    """Terminal color codes that work on Windows (via ANSI) and Unix."""

    def __init__(self, enabled=True):
        self.enabled = enabled

    def _c(self, code: int, text: str) -> str:
        if self.enabled:
            return f"\033[{code}m{text}\033[0m"
        return text

    def bold(self, s):    return self._c(1, s)
    def dim(self, s):     return self._c(2, s)
    def red(self, s):     return self._c(91, s)
    def green(self, s):   return self._c(92, s)
    def yellow(self, s):  return self._c(93, s)
    def blue(self, s):    return self._c(94, s)
    def magenta(self, s): return self._c(95, s)
    def cyan(self, s):    return self._c(96, s)
    def white(self, s):   return self._c(97, s)

    GRADIENT = [196, 202, 208, 214, 220, 226, 190, 154, 118, 82, 46]
    def gradient(self, s, level=0):
        """渐变色"""
        if self.enabled:
            code = self.GRADIENT[min(level, len(self.GRADIENT) - 1)]
            return f"\033[38;5;{code}m{s}\033[0m"
        return s


C = Colors()


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════
def format_size(num_elements, element_size=4):
    """将参数量转换为人类可读的内存大小"""
    bytes_total = num_elements * element_size
    if bytes_total < 1024:
        return f"{bytes_total} B"
    elif bytes_total < 1024**2:
        return f"{bytes_total/1024:.1f} KB"
    elif bytes_total < 1024**3:
        return f"{bytes_total/(1024**2):.2f} MB"
    else:
        return f"{bytes_total/(1024**3):.2f} GB"


def format_number(n):
    """添加千分位分隔符"""
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def safe_prod(shape):
    """安全计算 shape 乘积"""
    if len(shape) == 0:
        return 1
    p = 1
    for s in shape:
        p *= s
    return p


# ═══════════════════════════════════════════════════════════════
# 核心分析函数
# ═══════════════════════════════════════════════════════════════
def analyze_checkpoint(filepath):
    """完整分析 .pth 文件，返回结构化数据"""
    print(f"\n{C.bold('🔍 加载权重文件:')} {C.cyan(filepath)}")
    print(f"   文件大小: {C.green(format_size(os.path.getsize(filepath), 1))}\n")

    obj = torch.load(filepath, map_location='cpu', weights_only=False)

    result = {
        'type': type(obj).__name__,
        'filepath': filepath,
        'raw': obj,
    }

    if isinstance(obj, dict):
        result['num_keys'] = len(obj)
        result['top_keys'] = list(obj.keys())
        result['is_checkpoint'] = _is_training_checkpoint(obj)

        # 找到模型权重的位置
        model_dict = _extract_model_state(obj)
        result['model_dict'] = model_dict
        result['num_layers'] = len(model_dict)
        result['tree'] = _build_tree(model_dict)
        result['stats'] = _compute_stats(model_dict)
        result['extra_keys'] = [k for k in obj if k not in model_dict and k != 'model']

    elif isinstance(obj, (nn.Module,)):
        state = obj.state_dict()
        result['model_dict'] = state
        result['num_layers'] = len(state)
        result['tree'] = _build_tree(state)
        result['stats'] = _compute_stats(state)
        result['is_checkpoint'] = False

    return result


def _is_training_checkpoint(state_dict):
    """检测是否包含训练状态的完整 checkpoint"""
    training_keys = {'optimizer', 'lr_scheduler', 'epoch', 'optimizer_dict', 'scheduler'}
    overlap = set(state_dict.keys()) & training_keys
    return len(overlap) >= 2  # 至少包含 optimizer + epoch


def _extract_model_state(state_dict):
    """从各种格式中提取模型权重字典"""
    # 格式 1: {'model': OrderedDict(...), 'optimizer': ..., 'epoch': ...}
    if 'model' in state_dict and isinstance(state_dict['model'], dict):
        return state_dict['model']

    # 格式 2: {'state_dict': OrderedDict(...), ...}  (常见于分类模型)
    if 'state_dict' in state_dict and isinstance(state_dict['state_dict'], dict):
        return state_dict['state_dict']

    # 格式 3: 直接就是 weight dict
    # 检测第一个 key 是不是像 'backbone.xxx' 这样的模型层名
    if len(state_dict) > 0:
        first_key = next(iter(state_dict))
        if '.' in first_key or isinstance(state_dict[first_key], torch.Tensor):
            return {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}

    return {}


def _build_tree(param_dict):
    """按层级构建树结构 {prefix: {subtree | list_of_keys}}"""
    tree = {}
    for key in sorted(param_dict.keys()):
        parts = key.split('.')
        node = tree
        for i, part in enumerate(parts[:-1]):
            if part not in node:
                node[part] = {}
            node = node[part]
        # 叶子节点
        leaf_name = parts[-1]
        if '_leaves' not in node:
            node['_leaves'] = []
        node['_leaves'].append(leaf_name)
    return tree


def _compute_stats(param_dict):
    """计算统计信息"""
    total_params = 0
    total_elements = 0
    module_params = defaultdict(int)
    dtype_counts = defaultdict(int)
    min_shape = None
    max_shape = None
    min_shape_key = ''
    max_shape_key = ''

    for key, tensor in param_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        n = tensor.numel()
        total_params += n
        total_elements += n

        # 按模块统计
        module = key.split('.')[0] if '.' in key else 'root'
        module_params[module] += n

        # dtype 分布
        dtype_counts[str(tensor.dtype).split('.')[-1]] += 1

        # 最小/最大 shape
        if min_shape is None or n < safe_prod(min_shape):
            min_shape = tuple(tensor.shape)
            min_shape_key = key
        if max_shape is None or n > safe_prod(max_shape):
            max_shape = tuple(tensor.shape)
            max_shape_key = key

    return {
        'total_params': total_params,
        'total_elements': total_elements,
        'memory_mb': total_elements * 4 / (1024**2),
        'module_params': dict(module_params),
        'dtype_counts': dict(dtype_counts),
        'min_shape': (min_shape_key, min_shape),
        'max_shape': (max_shape_key, max_shape),
        'num_unique_tensors': len(param_dict),
    }


# ═══════════════════════════════════════════════════════════════
# 可视化输出
# ═══════════════════════════════════════════════════════════════

# Unicode box-drawing characters
TREE_BRANCH = "├── "
TREE_LAST  = "└── "
TREE_PIPE  = "│   "
TREE_SPACE = "    "


def print_header(title, width=78):
    print(f"\n{C.bold('━' * width)}")
    print(C.bold(f"  {title}"))
    print(C.bold('━' * width))


def print_overview(result):
    """打印总体概览"""
    print_header("📦 文件概览")

    print(f"  {C.dim('文件路径:')}  {result['filepath']}")
    print(f"  {C.dim('数据格式:')}  {C.yellow(result['type'])}")

    if result.get('is_checkpoint') is not None:
        if result['is_checkpoint']:
            print(f"  {C.dim('类型:')}      {C.green('✓ 完整训练 checkpoint (含 optimizer/epoch 等)')}")
        else:
            print(f"  {C.dim('类型:')}      {C.cyan('模型权重文件 (仅 state_dict)')}")

    if 'top_keys' in result:
        print(f"\n  {C.dim('顶层 keys:')}")
        for i, k in enumerate(result['top_keys']):
            is_model = (k == 'model') or (k == 'state_dict')
            marker = C.green(' ← 模型权重') if is_model else ''
            print(f"    {i+1:>2}. {C.yellow(k)}{marker}")

    if 'extra_keys' in result and result['extra_keys']:
        print(f"\n  {C.dim('其他 keys (非模型权重):')}")
        for k in result['extra_keys']:
            v = result['raw'].get(k)
            if isinstance(v, torch.Tensor):
                info = f"Tensor{list(v.shape)}"
            elif isinstance(v, (int, float)):
                info = f"{v}"
            else:
                info = type(v).__name__
            print(f"    - {C.yellow(k)}: {C.dim(info)}")  # 修复中文括号到英文


def print_stats(result):
    """打印统计信息"""
    print_header("📊 参数统计")
    stats = result['stats']

    print(f"  {C.dim('总参数量:')}    {C.green(format_number(stats['total_params']))} ({C.cyan(format_size(stats['total_params']))})")
    print(f"  {C.dim('总元素数:')}    {format_number(stats['total_elements'])}")
    mem_mb = stats['memory_mb']
    print(f"  {C.dim('内存占用 (fp32):')} {C.yellow(f'{mem_mb:.2f} MB')}")
    print(f"  {C.dim('唯一 tensor 数:')} {stats['num_unique_tensors']}")

    # dtype 分布
    if stats['dtype_counts']:
        print(f"\n  {C.dim('数据类型分布:')}")
        for dtype, count in sorted(stats['dtype_counts'].items(), key=lambda x: -x[1]):
            bar_width = min(30, count * 30 // max(stats['dtype_counts'].values()))
            bar = '█' * bar_width
            print(f"    {C.cyan(dtype):>16s}  {bar} {count}")

    # 按模块参数分布
    module_params = stats.get('module_params', {})
    if module_params:
        print(f"\n  {C.dim('模块参数分布:')}")
        total = stats['total_params']
        max_name_len = max(len(k) for k in module_params)
        for mod, n in sorted(module_params.items(), key=lambda x: -x[1]):
            pct = 100 * n / total
            bar_width = int(pct * 30 / 100)
            bar = '█' * bar_width
            print(f"    {C.magenta(mod+':').ljust(max_name_len+2)} {bar} {C.cyan(format_number(n))} ({pct:.1f}%)")

    # 最大/最小 tensor
    if stats['min_shape']:
        k, s = stats['min_shape']
        print(f"\n  {C.dim('最小 tensor:')} {C.yellow(k)} {C.dim(str(s))} ({safe_prod(s)} elements)")
    if stats['max_shape']:
        k, s = stats['max_shape']
        print(f"  {C.dim('最大 tensor:')} {C.yellow(k)} {C.dim(str(s))} ({format_number(safe_prod(s))} elements)")


def print_tree(result, show_detail=False):
    """递归打印模块树结构"""

    def _print_node(node, name, prefix="", is_last=True, depth=0):
        # 确定连接符
        conn = TREE_LAST if is_last else TREE_BRANCH

        # 收集统计信息
        leaves = node.get('_leaves', [])
        sub_modules = {k: v for k, v in node.items() if k != '_leaves'}

        total_params = 0
        for leaf in leaves:
            total_params += 1

        for sub in sub_modules.values():
            total_params += _count_tensors(sub)

        # 颜色
        if depth == 0:
            node_display = C.gradient(name, depth)
        elif depth <= 2:
            node_display = C.cyan(name)
        elif depth <= 4:
            node_display = C.green(name)
        else:
            node_display = C.dim(name)
        extra = f"  {C.dim(f'[{total_params} tensors]')}" if total_params > 0 else ""
        print(f"{prefix}{conn}{node_display}{extra}")

        # 计算下一级前缀
        if is_last:
            next_prefix = prefix + TREE_SPACE
        else:
            next_prefix = prefix + TREE_PIPE

        # 打印叶子节点
        if show_detail and leaves:
            for idx, leaf in enumerate(leaves):
                is_leaf_last = (idx == len(leaves) - 1) and not sub_modules
                lc = TREE_LAST if is_leaf_last else TREE_BRANCH
                print(f"{next_prefix}{lc}{C.dim(leaf)}")

        elif not show_detail and leaves:
            # 压缩显示
            shown = ', '.join(leaves[:5])
            more = f" (+{len(leaves)-5} more...)" if len(leaves) > 5 else ""
            print(f"{next_prefix}{TREE_LAST}{C.dim(shown)}{more}")

        # 递归打印子模块
        sub_items = list(sub_modules.items())
        for idx, (sub_name, sub_node) in enumerate(sub_items):
            sub_is_last = (idx == len(sub_items) - 1)
            _print_node(sub_node, sub_name, next_prefix, sub_is_last, depth + 1)

    def _count_tensors(node):
        leaves = len(node.get('_leaves', []))
        for k, v in node.items():
            if k != '_leaves':
                leaves += _count_tensors(v)
        return leaves

    print_header("🌲 模型权重树形结构")

    tree = result['tree']
    items = list(tree.items())
    for idx, (root_name, root_node) in enumerate(items):
        is_last = (idx == len(items) - 1)
        _print_node(root_node, root_name, "", is_last)


def print_detail(result):
    """打印每层的详细信息"""
    print_header("📋 逐层详细信息")

    param_dict = result.get('model_dict', {})
    if not param_dict:
        print(f"  {C.yellow('无模型权重可显示')}")
        return

    # 表格头
    header = f"  {'No.':>5s} │ {'Layer Name':<60s} │ {'Shape':<24s} │ {'Params':>10s} │ {'Memory':>10s}"
    sep = f"  {'─'*5}─┼─{'─'*60}─┼─{'─'*24}─┼─{'─'*10}─┼─{'─'*10}"
    print(C.bold(header))
    print(C.dim(sep))

    total_params = 0
    for i, (key, tensor) in enumerate(sorted(param_dict.items()), 1):
        if not isinstance(tensor, torch.Tensor):
            continue
        shape = list(tensor.shape)
        n = tensor.numel()
        total_params += n
        mem = format_size(n)

        # 截断过长的名字
        display_key = key if len(key) <= 58 else "..." + key[-55:]

        # 给不同类型的层着色
        if 'weight' in key.rsplit('.', 1)[-1]:
            line_color = C.green
        elif 'bias' in key.rsplit('.', 1)[-1]:
            line_color = C.yellow
        elif 'running_mean' in key or 'running_var' in key:
            line_color = C.magenta
        else:
            line_color = C.dim

        shape_str = str(shape)
        if len(shape_str) > 22:
            shape_str = shape_str[:19] + "..." + shape_str[-2:]

        row = (
            f"  {C.dim(str(i)):>5s} │ "
            f"{line_color(display_key):<60s} │ "
            f"{C.cyan(shape_str):<24s} │ "
            f"{C.green(format_number(n)):>10s} │ "
            f"{C.yellow(mem):>10s}"
        )
        print(row)

    # 合计行
    print(C.dim(sep))
    print(C.bold(
        f"  {'Total':>5s} │ {'─':>60s} │ {'─':>24s} │ "
        f"{C.green(format_number(total_params)):>10s} │ "
        f"{C.yellow(format_size(total_params)):>10s}"
    ))
    print()


def print_compare(result):
    """（可选）与当前工程模型对比"""
    print_header("🔄 模型结构对比（与当前默认配置构建的模型）")

    # 尝试构建当前工程的 DETR 模型
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from models import build_model

        # 尝试推测 num_classes
        param_dict = result.get('model_dict', {})
        if 'class_embed.weight' in param_dict:
            pretrained_nc = param_dict['class_embed.weight'].shape[0] - 1
        else:
            pretrained_nc = 91

        dummy_args = argparse.Namespace(
            lr=1e-4, lr_backbone=1e-5, batch_size=2, weight_decay=1e-4,
            epochs=300, lr_drop=200, clip_max_norm=0.1,
            frozen_weights=None, backbone='resnet50', dilation=False,
            position_embedding='sine', enc_layers=6, dec_layers=6,
            dim_feedforward=2048, hidden_dim=256, dropout=0.1,
            nheads=8, num_queries=100, pre_norm=False,
            masks=False, aux_loss=True, set_cost_class=1,
            set_cost_bbox=5, set_cost_giou=2,
            dice_loss_coef=1, bbox_loss_coef=5, giou_loss_coef=2,
            eos_coef=0.1, num_classes=pretrained_nc,
            dataset_file='coco', device='cpu',
        )
        current_model, _, _ = build_model(dummy_args)
        current_state = current_model.state_dict()

        # 对比
        pretrained_keys = set(param_dict.keys())
        current_keys = set(current_state.keys())

        missing = current_keys - pretrained_keys
        unexpected = pretrained_keys - current_keys
        common = current_keys & pretrained_keys

        shape_mismatch = []
        for key in sorted(common):
            ps = list(param_dict[key].shape)
            cs = list(current_state[key].shape)
            if ps != cs:
                shape_mismatch.append((key, ps, cs))

        shape_match = len(common) - len(shape_mismatch)

        print(f"  {C.dim('预训练权重层数:')}   {C.cyan(len(pretrained_keys))}")
        print(f"  {C.dim('当前模型层数:')}     {C.cyan(len(current_keys))}")
        print(f"  {C.dim('匹配 (shape 一致):')} {C.green(str(shape_match))}")
        print(f"  {C.dim('缺失 (新模型有/预训练无):')} {C.yellow(str(len(missing)))}")
        print(f"  {C.dim('多余 (预训练有/新模型无):')} {C.red(str(len(unexpected)))}")
        print(f"  {C.dim('shape 不匹配:')}   {C.magenta(str(len(shape_mismatch)))}")

        if shape_mismatch:
            print(f"\n  {C.bold('⚠️  Shape 不匹配的层:')}")
            sep = f"  {'─'*5}─┼─{'─'*50}─┼─{'─'*24}─┼─{'─'*24}"
            print(C.dim(sep))
            print(C.bold(f"  {'No.':>5s} │ {'Layer':<50s} │ {'Pretrained Shape':>24s} │ {'Current Shape':>24s}"))
            print(C.dim(sep))
            for i, (key, ps, cs) in enumerate(shape_mismatch, 1):
                print(f"  {i:>5d} │ {C.yellow(key):<50s} │ {C.red(str(ps)):>24s} │ {C.green(str(cs)):>24s}")
            print(C.dim(sep))

        if missing:
            print(f"\n  {C.bold('缺少的层 (前 20 个):')}")
            for key in sorted(missing)[:20]:
                print(f"    {C.yellow('- '+key)} {C.dim(str(list(current_state[key].shape)))}")
            if len(missing) > 20:
                print(f"    {C.dim(f'... 还有 {len(missing)-20} 个')}")

        if unexpected:
            print(f"\n  {C.bold('多余的层 (前 20 个):')}")
            for key in sorted(unexpected)[:20]:
                print(f"    {C.red('+ '+key)} {C.dim(str(list(param_dict[key].shape)))}")
            if len(unexpected) > 20:
                print(f"    {C.dim(f'... 还有 {len(unexpected)-20} 个')}")

    except ImportError as e:
        print(f"  {C.yellow('无法构建当前模型进行对比:')} {e}")
        print(f"  {C.dim('请确保在项目根目录运行，且依赖已安装。')}")
    except Exception as e:
        print(f"  {C.red('对比失败:')} {e}")


def print_class_embed_info(result):
    """额外输出 class_embed / bbox_embed 详细信息（DETR 专属）"""
    param_dict = result.get('model_dict', {})
    head_keys = sorted([k for k in param_dict if 'class_embed' in k or 'bbox_embed' in k])

    if not head_keys:
        return

    print_header("🎯 检测头信息 (Detection Head)")
    for key in head_keys:
        tensor = param_dict[key]
        shape = list(tensor.shape)
        n = tensor.numel()
        print(f"  {C.yellow(key):<40s}  shape={C.cyan(str(shape)):<20s}  "
              f"params={C.green(format_number(n)):<12s}  "
              f"dtype={C.dim(str(tensor.dtype).split('.')[-1])}")

    # class_embed 类别推断
    if 'class_embed.weight' in param_dict:
        num_classes = param_dict['class_embed.weight'].shape[0] - 1
        print(f"\n  {C.dim('推断类别数:')} {C.green(str(num_classes))}  (class_embed 输出维度 - 1 = 背景类)")

        # 打印每个类别的权重统计
        w = param_dict['class_embed.weight']
        print(f"\n  {C.dim('类别权重统计:')}")
        for i in range(w.shape[0] - 1):
            label = f"class_{i}"
            mean_v = w[i].mean().item()
            std_v = w[i].std().item()
            print(f"    {C.cyan(label):<20s} mean={mean_v:+.6f}  std={std_v:.6f}")
        # 背景类
        bg_idx = w.shape[0] - 1
        bg_mean = w[bg_idx].mean().item()
        bg_std = w[bg_idx].std().item()
        print(f"    {C.magenta(f'background (idx {bg_idx})'):<20s} mean={bg_mean:+.6f}  std={bg_std:.6f}")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='分析 .pth / .pt 权重文件的结构和内容',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python analysispt.py --weights detr-r50.pth
  python analysispt.py --weights detr-r50.pth --detail --compare
  python analysispt.py --weights my_checkpoint.pth --compare
  python analysispt.py --weights my_checkpoint.pth --detail --compact
        """
    )
    parser.add_argument('--weights', '-w', type=str, required=True,
                        help='权重文件路径（.pth / .pt）')
    parser.add_argument('--detail', '-d', action='store_true',
                        help='显示每层 tensor 的详细信息（shapes, 参数量等）')
    parser.add_argument('--compare', '-c', action='store_true',
                        help='与当前工程构建的 DETR 模型进行对比')
    parser.add_argument('--no-color', action='store_true',
                        help='禁用彩色输出')
    parser.add_argument('--compact', action='store_true',
                        help='紧凑模式：只显示概览和统计，不显示树结构')

    args = parser.parse_args()

    # 颜色设置
    global C
    C = Colors(enabled=not args.no_color)

    # Windows 启用 ANSI 转义
    if not args.no_color and sys.platform == 'win32':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass

    if not os.path.exists(args.weights):
        print(f"{C.red('✗ 文件不存在:')} {args.weights}")
        sys.exit(1)

    # ══════════ 分析 ══════════
    result = analyze_checkpoint(args.weights)

    # ══════════ 输出 ══════════
    print_overview(result)
    print_stats(result)

    if not args.compact:
        print_tree(result, show_detail=args.detail)

    if args.detail:
        print_detail(result)

    print_class_embed_info(result)

    if args.compare:
        print_compare(result)

    # 最终总结
    print(f"\n{C.bold('━' * 78)}")
    print(f"  {C.green('✓ 分析完成')}")
    print(f"{C.bold('━' * 78)}\n")


if __name__ == '__main__':
    main()
