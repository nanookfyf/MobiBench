import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import os
import math
import random

def find_video_files(base_dir: Path, min_frames: int = 10, max_videos: Optional[int] = None) -> List[Path]:
    """
    查找所有视频文件，并筛选帧数
    min_frames: 最小帧数要求
    max_videos: 最大视频数量限制（None表示不限制）
    """
    video_files = []
    valid_videos = []
    
    for root, dirs, files in os.walk(base_dir):
        root_path = Path(root)
        for file in files:
            if file.endswith('.mp4') and "video" in file.lower():
                video_path = root_path / file
                
                # 检查视频帧数
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    
                    if frame_count >= min_frames:
                        video_files.append((video_path, frame_count, fps))
    
    # 按帧数降序排序（帧数多的在前）
    video_files.sort(key=lambda x: x[1], reverse=True)
    
    # 限制数量
    if max_videos and len(video_files) > max_videos:
        video_files = video_files[:max_videos]
    
    # 只返回路径
    valid_videos = [item[0] for item in video_files]
    
    # 打印统计信息
    if video_files:
        total_frames = sum(item[1] for item in video_files)
        avg_frames = total_frames / len(video_files)
        print(f"[INFO] 找到 {len(video_files)} 个有效视频 (>= {min_frames}帧)")
        print(f"[INFO] 帧数范围: {video_files[-1][1]} - {video_files[0][1]} 帧")
        print(f"[INFO] 平均帧数: {avg_frames:.1f} 帧")
        print(f"[INFO] 总帧数: {total_frames} 帧")

    return random.sample(valid_videos, k=len(valid_videos)-1) if valid_videos else []

def create_video_matrix(video_paths: List[Path], output_path: Path, 
                       cols: int = 3, video_size: Tuple[int, int] = (400, 300),
                       padding: int = 10, bg_color: Tuple[int, int, int] = (30, 30, 30)):
    """
    创建视频矩阵（精简版）
    """
    if not video_paths:
        print("[ERROR] 没有找到视频文件")
        return
    
    print(f"[INFO] 处理 {len(video_paths)} 个视频")
    
    # 打开所有视频
    caps = []
    video_frames = []
    max_frames = 0
    
    for vpath in video_paths:
        cap = cv2.VideoCapture(str(vpath))
        if cap.isOpened():
            caps.append(cap)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_frames.append(frame_count)
            max_frames = max(max_frames, frame_count)
            print(f"  {vpath.stem[:20]:20s} {frame_count:4d}帧")
        else:
            print(f"  [WARN] 无法打开: {vpath}")
            caps.append(None)
            video_frames.append(0)
    
    if not caps:
        print("[ERROR] 没有可用的视频")
        return
    
    # 计算布局
    n_videos = len(caps)
    rows = (n_videos + cols - 1) // cols
    
    # 计算画布尺寸
    video_width, video_height = video_size
    canvas_width = (video_width + padding) * cols + padding
    canvas_height = (video_height + padding) * rows + padding
    
    print(f"[INFO] 布局: {rows}行 × {cols}列")
    print(f"[INFO] 画布: {canvas_width}x{canvas_height}")
    print(f"[INFO] 最长视频: {max_frames}帧")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 3.0, (canvas_width, canvas_height))
    
    if not out.isOpened():
        print("[ERROR] 无法创建输出视频")
        for cap in caps:
            if cap:
                cap.release()
        return
    
    # 处理每一帧
    for frame_idx in range(max_frames):
        # 创建画布
        canvas = np.full((canvas_height, canvas_width, 3), bg_color, dtype=np.uint8)
        
        # 处理每个视频
        for i, cap in enumerate(caps):
            row = i // cols
            col = i % cols
            
            x = padding + col * (video_width + padding)
            y = padding + row * (video_height + padding)
            
            # 读取帧
            frame = None
            if cap and cap.isOpened() and frame_idx < video_frames[i]:
                ret, frame = cap.read()
            
            # 如果视频已结束，使用最后一帧
            if frame is None and frame_idx >= video_frames[i] and video_frames[i] > 0:
                # 重新打开视频获取最后一帧
                temp_cap = cv2.VideoCapture(str(video_paths[i]))
                if temp_cap.isOpened():
                    # 跳转到最后一帧
                    last_frame = video_frames[i] - 1
                    temp_cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame)
                    ret, frame = temp_cap.read()
                    temp_cap.release()
            
            if frame is not None:
                # 调整尺寸
                resized = cv2.resize(frame, (video_width, video_height))
                canvas[y:y+video_height, x:x+video_width] = resized
                
                # 边框
                border_color = (0, 200, 0) if frame_idx < video_frames[i] else (100, 100, 100)
                cv2.rectangle(canvas,
                            (x, y),
                            (x + video_width - 1, y + video_height - 1),
                            border_color, 2)
                
                # 视频编号
                cv2.putText(canvas, f"#{i+1}",
                           (x + 10, y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (255, 255, 255), 2)
        
        # 写入帧
        out.write(canvas)
    
    # 所有视频播放完后，停留在最后一帧
    # 获取最后一帧（所有视频都结束时的画布）
    #for _ in range(30):  # 停留3秒（10fps × 3秒）
    #    out.write(canvas)
    
    # 释放资源
    for cap in caps:
        if cap:
            cap.release()
    out.release()
    
    print(f"[INFO] 完成: {output_path}")

def generate_multiple_matrices(video_paths: List[Path], output_dir: Path,
                               duration_seconds: Optional[float] = None,
                               video_size: Tuple[int, int] = (150, 330),
                               padding: int = 5,
                               bg_color: Tuple[int, int, int] = (30, 30, 30),
                               fps: float = 3.0):
    """
    自动生成多个不同列数的视频矩阵
    """
    if not video_paths:
        print("[ERROR] 没有视频文件")
        return
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 列数配置
    col_configs = [
        
        #{"cols": 5, "video_size": (200, 440)},    # 5列，中等
        #{"cols": 8, "video_size": (200, 440)},    # 8列，小一点
        #{"cols": 10, "video_size": (200, 440)},   # 10列，更小
        #{"cols": 15, "video_size": (200, 440)},    # 15列，最小
        #{"cols": 3, "video_size": (250, 550)},    # 3列，大一点
        {"cols": 20, "video_size": (220, 440)},    # 4列，适中
    ]
    
    # 根据视频数量动态调整列数
    n_videos = len(video_paths)
    print(f"[INFO] 总视频数: {n_videos}")
    
    # 为每个列数配置生成视频
    for config in col_configs:
        cols = config["cols"]
        
        # 如果列数太多导致视频太小，跳过
        if cols > 20:  # 最大列数限制
            continue
            
        # 计算需要的行数
        rows_needed = math.ceil(n_videos / cols)
        
        # 如果行数太多（超过10行），说明视频太小，跳过
        # if rows_needed > 10:
        #     print(f"[INFO] 跳过 {cols}列: 需要{rows_needed}行，视频太小")
        #     continue
        
        # 生成输出文件名
        output_filename = f"video_matrix_{cols}cols_{n_videos}videos.mp4"
        output_path = output_dir / output_filename
        
        print(f"\n{'='*60}")
        print(f"[INFO] 生成 {cols}列视频矩阵")
        print(f"[INFO] 视频尺寸: {config['video_size']}")
        print(f"[INFO] 布局: {cols}行 × {cols}列")
        print(f"[INFO] 输出文件: {output_path.name}")
        
        # 创建视频矩阵
        success = create_video_matrix(
            video_paths=video_paths[:cols * cols],  # 只取足够的视频
            output_path=output_path,
            cols=cols,
            #duration_seconds=duration_seconds,
            video_size=config["video_size"],
            padding=padding,
            bg_color=bg_color,
            #fps=fps
        )
        
        if not success:
            print(f"[ERROR] 生成 {cols}列视频失败")


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="自动生成多个不同列数的视频矩阵")
    parser.add_argument("--input", type=str, required=True, help="输入目录")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--min-frames", type=int, default=10, help="最小帧数要求")
    parser.add_argument("--max-videos", type=int, default=None, help="最大视频数量限制")
    parser.add_argument("--duration", type=float, default=4.0, help="视频时长（秒）")
    parser.add_argument("--fps", type=float, default=3.0, help="输出视频帧率")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        return
    
    # 查找并筛选视频文件
    print(f"[INFO] 搜索视频文件，最小帧数: {args.min_frames}")
    video_files = find_video_files(
        base_dir=input_dir,
        min_frames=args.min_frames,
        max_videos=args.max_videos
    )
    
    if not video_files:
        print("没有找到符合条件的视频文件")
        return
    
    print(f"\n[INFO] 开始生成多个视频矩阵...")
    print(f"[INFO] 目标时长: {args.duration}秒")
    print(f"[INFO] 输出帧率: {args.fps}fps")
    print(f"[INFO] 输出目录: {output_dir}")
    
    # 自动生成多个不同列数的视频
    generate_multiple_matrices(
        video_paths=video_files,
        output_dir=output_dir,
        duration_seconds=args.duration,
        video_size=(150, 330),
        padding=5,
        bg_color=(30, 30, 30),
        fps=args.fps
    )
    
    print(f"\n{'='*60}")
    print("[INFO] 所有视频矩阵生成完成！")


if __name__ == "__main__":
    main()