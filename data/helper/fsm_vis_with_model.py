#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model_path_annotate.py

功能：
- 读取某一次 UI-TARS 离线评估目录下的 fsm_eval_trace.json
- 按 step 顺序取出每一步的截图，按顺序排成一张大图，用箭头连接
- 对每张图：
    * 用 OmniParser 画出所有 UI 元素框（红色）
    * 如果该图上有模型 click 操作，在图上画一个醒目的绿色实心圆点

使用示例：
python model_path_annotate.py ^
  --run_dir D:\cdl\code\MobiBench\runs\bilibili\type2\20251124_134332_在b站搜索starcraft并播放相关内容的第一个视频 ^
  --output D:\cdl\code\MobiBench\runs\bilibili_type2_starcraft_path.png
"""

import os
import json
import math
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from MobiBench.utils.parse_omni import extract_all_bounds

def load_steps(run_dir: Path):
    """
    从 run_dir/fsm_eval_trace.json 读取 step 信息
    返回：
      images: 按时间顺序的图片路径列表 [img0, img1, img2, ...]
      clicks_by_img: { img_path -> [(x,y), ...] }
    """
    trace_path = run_dir / "fsm_eval_trace.json"
    if not trace_path.is_file():
        raise FileNotFoundError(f"找不到 {trace_path}")

    with open(trace_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = data.get("steps", [])
    # 按 step 字段排序（防止顺序乱）
    steps = sorted(steps, key=lambda s: s.get("step", 0))

    if not steps:
        raise RuntimeError("fsm_eval_trace.json 里没有 steps")

    images = []
    # 从第一步的 prev_img 开始
    images.append(steps[0].get("prev_img"))

    for s in steps:
        images.append(s.get("new_img"))

    # 收集每张图上的 click 坐标（以 prev_img 为准）
    clicks_by_img = {}
    for s in steps:
        if s.get("action_type") != "click":
            continue
        img = s.get("prev_img")
        ap = s.get("action_params", {}) or {}
        cx = ap.get("position_x")
        cy = ap.get("position_y")
        if img and cx is not None and cy is not None:
            clicks_by_img.setdefault(img, []).append((cx, cy))

    return images, clicks_by_img


def draw_arrow(draw: ImageDraw.ImageDraw,
               p0, p1,
               color=(0, 0, 0),
               width=6,
               head_len=40,
               head_width=28):
    """
    在大画布上画一条从 p0->p1 的带箭头直线
    """
    x0, y0 = p0
    x1, y1 = p1
    draw.line([p0, p1], fill=color, width=width)

    dx = x1 - x0
    dy = y1 - y0
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L

    # 箭头三角形三个点
    bx = x1 - head_len * ux
    by = y1 - head_len * uy

    left = (bx - head_width * uy / 2, by + head_width * ux / 2)
    right = (bx + head_width * uy / 2, by - head_width * ux / 2)

    draw.polygon([p1, left, right], fill=color)


def annotate_single_image(img_path: str,
                          boxes,
                          clicks,
                          tile_width: int):
    """
    对单张原图：
      - 缩放到 tile_width 宽度
      - 画出 boxes（红框）
      - 在 clicks 位置画绿色实心圆
    返回：处理过的 PIL.Image
    """
    im = Image.open(img_path).convert("RGB")
    w0, h0 = im.size
    scale = tile_width / float(w0)
    new_h = int(h0 * scale)
    im = im.resize((tile_width, new_h), Image.LANCZOS)

    draw = ImageDraw.Draw(im)

    # 画所有 UI 框（细红线）
    if boxes:
        for (x1, y1, x2, y2) in boxes:
            x1s = x1 * scale
            y1s = y1 * scale
            x2s = x2 * scale
            y2s = y2 * scale
            draw.rectangle([x1s, y1s, x2s, y2s], outline=(255, 0, 0), width=3)

    # 画点击点（绿色实心圆，比较大）
    if clicks:
        for (cx, cy) in clicks:
            cxs = cx * scale
            cys = cy * scale
            r = 18  # 半径大一点，明显一些
            draw.ellipse(
                [cxs - r, cys - r, cxs + r, cys + r],
                fill=(0, 255, 0),
                outline=(0, 0, 0),
                width=3,
            )

    return im


def build_big_canvas(images, clicks_by_img, output_path: str,
                     tile_width=600, margin=60, arrow_gap=120,
                     with_boxes=True):
    """
    把多张图排成一行，用箭头连接，保存到 output_path
    """
    # 先为每张图准备 boxes（如果有 parse_omni）
    unique_imgs = sorted(set([p for p in images if p]))
    boxes_cache = {}
    if with_boxes and extract_all_bounds is not None:
        for p in unique_imgs:
            try:
                boxes = extract_all_bounds(p)
            except Exception as e:
                print(f"[WARN] 提取 {p} 边界框失败：{e}")
                boxes = []
            boxes_cache[p] = boxes
    else:
        for p in unique_imgs:
            boxes_cache[p] = []

    # 对每张图做缩放+标注
    frames = []
    for p in images:
        if not p:
            continue
        boxes = boxes_cache.get(p, [])
        clicks = clicks_by_img.get(p, [])
        im = annotate_single_image(p, boxes, clicks, tile_width)
        frames.append((p, im))

    if not frames:
        raise RuntimeError("没有任何有效的图片可以绘制")

    max_h = max(im.height for _, im in frames)
    n = len(frames)

    canvas_w = margin * 2 + n * tile_width + (n - 1) * arrow_gap
    canvas_h = margin * 2 + max_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # 逐张贴图，并记录每张图的位置，画箭头用
    x = margin
    centers = []
    for _, im in frames:
        y = margin + (max_h - im.height) // 2
        canvas.paste(im, (x, y))
        cx = x + im.width // 2
        cy = y + im.height // 2
        centers.append((x, y, im.width, im.height, cx, cy))
        x += tile_width + arrow_gap

    # 画箭头：从每一张图的右中 -> 下一张图的左中
    for i in range(n - 1):
        x, y, w, h, _, cy = centers[i]
        x2, y2, _, h2, _, cy2 = centers[i + 1]
        start = (x + w, y + h // 2)
        end = (x2, y2 + h2 // 2)
        draw_arrow(draw, start, end, color=(0, 0, 0))

    # 保存
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out), dpi=(300, 300))
    print(f"[INFO] 已保存到 {out}")


def main():
    parser = argparse.ArgumentParser(
        description="把单次模型轨迹上的截图排成一张大图，标出所有 UI 框和模型点击点"
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="某次 UI-TARS 离线评估的目录，例如 D:/.../runs/bilibili/type2/20251124_xxx",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出图片路径，例如 D:/.../runs/bilibili_type2_path.png",
    )
    parser.add_argument(
        "--no-boxes",
        action="store_true",
        help="不要画 Omni 检测到的红框，只画点击点",
    )

    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    images, clicks_by_img = load_steps(run_dir)
    print(f"[INFO] 轨迹共 {len(images)} 帧，其中有 click 的帧数：{len(clicks_by_img)}")

    build_big_canvas(
        images=images,
        clicks_by_img=clicks_by_img,
        output_path=args.output,
        with_boxes=not args.no_boxes,
    )


if __name__ == "__main__":
    main()
