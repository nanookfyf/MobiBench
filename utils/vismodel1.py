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
- 支持生成轨迹图片和视频

使用示例：
# 生成单次轨迹的图片和视频
python model_path_annotate.py ^
  --run_dir D:\cdl\code\MobiBench\runs\bilibili\type2\20251124_134332_在b站搜索starcraft并播放相关内容的第一个视频 ^
  --output D:\cdl\code\MobiBench\runs\bilibili_type2_starcraft_path.png

# 批量处理
python model_path_annotate.py --run_dir /Users/fff/Desktop/mobiagent/MobiBench/agents/UI_TARS/runs
"""

import os
import json
import math
import argparse
from pathlib import Path
import cv2
import numpy as np
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
    print(f"[INFO] 轨迹图片已保存到 {out}")
    
    # 返回画布用于生成视频
    return canvas, frames, centers


def generate_simple_slideshow_video(frames, output_video_path: str, 
                                    fps=1.0, display_seconds=1.0, 
                                    final_display_seconds=2.0):
    """
    生成简单的幻灯片视频：一张图一张图播放，最后停留在最后一张图
    简单版本：不添加额外的完成标记
    """
    if not frames:
        print("[WARN] 没有图片可以生成视频")
        return
    
    # 检查所有图片尺寸是否一致
    img_widths = [img.width for _, img in frames]
    img_heights = [img.height for _, img in frames]
    
    # 如果图片尺寸不一致，统一到最大尺寸
    max_width = max(img_widths)
    max_height = max(img_heights)
    
    # 创建视频写入器
    frame_width, frame_height = max_width, max_height
    
    # 为了更好的兼容性，确保尺寸是偶数
    if frame_width % 2 != 0:
        frame_width += 1
    if frame_height % 2 != 0:
        frame_height += 1
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        output_video_path, 
        fourcc, 
        fps, 
        (frame_width, frame_height)
    )
    
    if not video_writer.isOpened():
        print(f"[ERROR] 无法创建视频写入器")
        return
    
    n = len(frames)
    
    print(f"[INFO] 开始生成幻灯片视频，共 {n} 张图片")
    
    # 计算帧数
    frames_per_image = int(fps * display_seconds)
    frames_final_image = int(fps * final_display_seconds)
    
    # 处理每张图片
    for idx, (img_path, pil_img) in enumerate(frames):
        print(f"[INFO] 处理第 {idx+1}/{n} 张图片")
        
        # 将PIL图像转换为OpenCV格式
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # 调整尺寸
        h, w = img_cv.shape[:2]
        if w != frame_width or h != frame_height:
            # 保持宽高比缩放
            scale = min(frame_width / w, frame_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_resized = cv2.resize(img_cv, (new_w, new_h))
            
            # 创建带白色边框的画布
            canvas = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
            x_offset = (frame_width - new_w) // 2
            y_offset = (frame_height - new_h) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
            img_cv = canvas
        
        # 添加步数信息
        text = f"{idx+1}/{n}"
        cv2.putText(img_cv, text,
                   (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1.2,
                   (0, 0, 0),
                   3)
        
        # 如果是最后一张图片，显示时间更长
        frames_to_write = frames_per_image
        if idx == n - 1:
            frames_to_write = frames_final_image
        
        # 写入帧
        for _ in range(frames_to_write):
            video_writer.write(img_cv)
    
    video_writer.release()
    print(f"[INFO] 视频已保存: {output_video_path}")

def generate_animation_video(frames, centers, output_video_path: str,
                             fps=10, tile_width=600, margin=60, arrow_gap=120):
    """
    生成更生动的动画视频，逐步显示每张图片
    """
    if not frames:
        return
    
    # 计算画布尺寸
    max_h = max(im.height for _, im in frames)
    n = len(frames)
    canvas_w = margin * 2 + n * tile_width + (n - 1) * arrow_gap
    canvas_h = margin * 2 + max_h
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (canvas_w, canvas_h))
    
    # 生成背景（白色）
    background = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    # 动画参数
    total_duration = n * 1.5  # 每张图片1.5秒
    total_frames = int(total_duration * fps)
    
    for frame_idx in range(total_frames):
        # 创建当前帧
        current_frame = background.copy()
        
        # 计算当前时间
        current_time = frame_idx / fps
        
        # 确定当前应该显示多少张图片
        images_to_show = min(n, int(current_time / 1.5) + 1)
        
        for i in range(images_to_show):
            x, y, w_img, h_img, _, _ = centers[i]
            _, img_pil = frames[i]
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            # 如果是最后一张图片，可能有淡入效果
            if i == images_to_show - 1:
                image_time = current_time - (i * 1.5)
                if image_time < 0.5:  # 淡入效果
                    alpha = min(1.0, image_time / 0.5)
                    roi = current_frame[y:y+h_img, x:x+w_img]
                    blended = cv2.addWeighted(img_cv, alpha, roi, 1-alpha, 0)
                    current_frame[y:y+h_img, x:x+w_img] = blended
                else:
                    current_frame[y:y+h_img, x:x+w_img] = img_cv
            else:
                current_frame[y:y+h_img, x:x+w_img] = img_cv
        
        # 绘制已连接的箭头
        for i in range(images_to_show - 1):
            x, y, w_img, h_img, _, _ = centers[i]
            x2, y2, w_img2, h_img2, _, _ = centers[i + 1]
            start = (x + w_img, y + h_img // 2)
            end = (x2, y2 + h_img2 // 2)
            
            # 绘制箭头
            cv2.arrowedLine(current_frame, start, end, 
                           color=(0, 0, 0), thickness=6,
                           tipLength=0.2)
        
        # 添加进度信息
        progress_text = f"Step {images_to_show}/{n}"
        cv2.putText(current_frame, progress_text, 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 
                    (0, 0, 0), 3)
        
        # 添加时间戳
        time_text = f"Time: {current_time:.1f}s"
        cv2.putText(current_frame, time_text, 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                    (100, 100, 100), 2)
        
        video_writer.write(current_frame)
    
    video_writer.release()
    print(f"[INFO] 动画视频已保存到 {output_video_path}")


def process_single_run(run_dir: Path, output_dir: Path, with_boxes=True, 
                       generate_video_flag=True, video_type="animation"):
    """
    处理单次运行的轨迹
    """
    try:
        images, clicks_by_img = load_steps(run_dir)
        print(f"[INFO] 处理 {run_dir.name}: 轨迹共 {len(images)} 帧")
        
        # 生成输出路径
        output_path = output_dir / f"{run_dir.name}_path.png"
        video_path = output_dir / f"{run_dir.name}_video.mp4"
        
        # 生成轨迹图片
        canvas, frames, centers = build_big_canvas(
            images=images,
            clicks_by_img=clicks_by_img,
            output_path=output_path,
            with_boxes=with_boxes,
        )
        
        # 生成视频
        if generate_video_flag and frames:
            if video_type == "animation":
                generate_animation_video(frames, centers, video_path)
            else:
                video_path = output_dir / f"{run_dir.name}_simple_video.mp4"
                generate_simple_slideshow_video(frames,  video_path)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 处理 {run_dir} 时出错: {e}")
        return False


def main():
    """
    主函数：支持单次运行或批量处理
    """
    parser = argparse.ArgumentParser(
        description="生成模型轨迹的可视化图片和视频"
    )
    
    parser.add_argument(
        "--run_dir",
        required=True,
        help="单个运行目录或包含多个运行目录的父目录"
    )
    
    parser.add_argument(
        "--output_dir",
        default=None,
        help="输出目录（默认为输入目录）"
    )
    
    parser.add_argument(
        "--no-boxes",
        action="store_true",
        help="不要画 Omni 检测到的红框，只画点击点"
    )
    
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="不生成视频，只生成图片"
    )
    
    parser.add_argument(
        "--video-type",
        choices=["simple", "animation"],
        default="animation",
        help="视频类型：simple（简单逐步显示）或 animation（动画效果）"
    )
    
    parser.add_argument(
        "--single",
        action="store_true",
        help="处理单个运行目录（而不是批量处理）"
    )
    
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = run_dir if args.single else run_dir / "visualizations"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.single:
        # 处理单个运行目录
        if not run_dir.is_dir():
            print(f"[ERROR] 目录不存在: {run_dir}")
            return
        
        process_single_run(
            run_dir=run_dir,
            output_dir=output_dir,
            with_boxes=not args.no_boxes,
            generate_video_flag=not args.no_video,
            video_type=args.video_type
        )
    else:
        # 批量处理：查找所有包含 fsm_eval_trace.json 的目录
        print(f"[INFO] 开始批量处理目录: {run_dir}")
        
        # 查找所有可能的运行目录
        run_dirs = []
        for root, dirs, files in os.walk(run_dir):
            root_path = Path(root)
            if "fsm_eval_trace.json" in files:
                run_dirs.append(root_path)
        
        print(f"[INFO] 找到 {len(run_dirs)} 个运行目录")
        
        # 处理每个目录
        success_count = 0
        for i, run_path in enumerate(run_dirs):
            print(f"[INFO] 正在处理 ({i+1}/{len(run_dirs)}): {run_path.relative_to(run_dir)}")
            
            # 为每个运行创建单独的输出子目录
            relative_path = run_path.relative_to(run_dir)
            run_output_dir = output_dir / relative_path
            run_output_dir.mkdir(parents=True, exist_ok=True)
            
            if process_single_run(
                run_dir=run_path,
                output_dir=run_output_dir,
                with_boxes=not args.no_boxes,
                generate_video_flag=not args.no_video,
                video_type=args.video_type
            ):
                success_count += 1
        
        print(f"[INFO] 批量处理完成！成功处理 {success_count}/{len(run_dirs)} 个目录")


if __name__ == "__main__":
    main()