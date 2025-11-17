# Just for testing visualization of detection results and clicked elements

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from omni_utils import get_som_labeled_img, check_ocr_box, get_yolo_model
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
detect_model_path='/Users/fengyunfei/Desktop/mobiagent/Mobibench/static_bench/models/weights/OminParserv2/icon_detect/model.pt'
caption_model_path='./weights/icon_caption_florence'

som_model = get_yolo_model(detect_model_path)
som_model.to(device)
def visualize_detection_results(screenshot_path, bounds_list, clicked_bounds=None, click_point=None):
    """可视化检测结果和边界框"""
    
    # 打开图片
    image = Image.open(screenshot_path).convert('RGB')
    image_array = np.array(image)
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(image_array)
    
    # 绘制所有检测到的边界框
    for i, bounds in enumerate(bounds_list):
        left, top, right, bottom = bounds
        width = right - left
        height = bottom - top
        
        # 创建矩形框
        rect = patches.Rectangle(
            (left, top), width, height,
            linewidth=2, edgecolor='red', facecolor='none', alpha=0.7
        )
        ax.add_patch(rect)
        
        # 添加编号
        ax.text(left, top-5, f'{i}', fontsize=8, color='red', 
                bbox=dict(boxstyle="round,pad=0.1", facecolor='red', alpha=0.7, edgecolor='none'))
    
    # 如果有点击的边界框，用不同颜色高亮显示
    if clicked_bounds is not None:
        left, top, right, bottom = clicked_bounds
        width = right - left
        height = bottom - top
        
        rect = patches.Rectangle(
            (left, top), width, height,
            linewidth=3, edgecolor='blue', facecolor='none', alpha=0.9
        )
        ax.add_patch(rect)
        ax.text(left, top-10, 'Clicked', fontsize=10, color='blue', 
                bbox=dict(boxstyle="round,pad=0.2", facecolor='blue', alpha=0.8, edgecolor='none'))
    
    # 如果有点击点，标记出来
    if click_point is not None:
        click_x, click_y = click_point
        ax.plot(click_x, click_y, 'go', markersize=8, markeredgewidth=2, 
                markerfacecolor='none', markeredgecolor='green', label='Click Point')
    
    ax.set_title(f'UI Element Detection Results - Total {len(bounds_list)} elements', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_parsed_details(parsed_content_list):
    """可视化解析的详细信息"""
    print("=" * 80)
    print("DETAILED PARSING RESULTS")
    print("=" * 80)
    
    for i, item in enumerate(parsed_content_list):
        print(f"\n--- Element {i} ---")
        print(f"Category: {item.get('category', 'N/A')}")
        print(f"Text: {item.get('text', 'N/A')}")
        
        bbox = item.get('bbox', [])
        if bbox and len(bbox) >= 4:
            print(f"BBox (ratio): [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]")
        
        # 如果有其他重要信息也打印出来
        for key, value in item.items():
            if key not in ['bbox', 'category', 'text'] and value:
                print(f"{key}: {value}")

def save_detection_image(screenshot_path, bounds_list, output_path, clicked_bounds=None):
    """保存带边界框标注的图片"""
    
    image = Image.open(screenshot_path).convert('RGB')
    image_array = np.array(image)
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(image_array)
    
    # 绘制所有边界框
    for i, bounds in enumerate(bounds_list):
        left, top, right, bottom = bounds
        width = right - left
        height = bottom - top
        
        rect = patches.Rectangle(
            (left, top), width, height,
            linewidth=2, edgecolor='red', facecolor='none', alpha=0.7
        )
        ax.add_patch(rect)
        ax.text(left, top-5, f'{i}', fontsize=8, color='red',
                bbox=dict(boxstyle="round,pad=0.1", facecolor='red', alpha=0.7))
    
    # 高亮点击的边界框
    if clicked_bounds is not None:
        left, top, right, bottom = clicked_bounds
        width = right - left
        height = bottom - top
        
        rect = patches.Rectangle(
            (left, top), width, height,
            linewidth=3, edgecolor='blue', facecolor='none', alpha=0.9
        )
        ax.add_patch(rect)
        ax.text(left, top-10, 'Clicked', fontsize=10, color='blue',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='blue', alpha=0.8))
    
    ax.set_title(f'UI Element Detection - {len(bounds_list)} elements', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Detection result saved to: {output_path}")

# 修改你的主函数，加入可视化功能
def extract_all_bounds_with_visualization(screenshot_path, visualize=True):
    """提取边界框并可选地进行可视化"""
    image = Image.open(screenshot_path).convert('RGB')
    
    # OCR检测文本框
    (text, ocr_bbox), _ = check_ocr_box(
        image,
        display_img=False, 
        output_bb_format='xyxy', 
        easyocr_args={'text_threshold': 0.9}, 
        use_paddleocr=True,
    )

    # YOLO检测UI元素
    _, _, parsed_content_list = get_som_labeled_img(
        image, 
        som_model, 
        BOX_TRESHOLD=0.1, 
        output_coord_in_ratio=True, 
        ocr_bbox=ocr_bbox,
        ocr_text=text,
        use_local_semantics=False,
        iou_threshold=0.7, 
        scale_img=False
    )

    # 提取边界框并转换为绝对坐标
    image_width, image_height = image.size
    bounds_list = []

    for item in parsed_content_list:
        bbox = item.get('bbox')
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            # 转换为绝对坐标
            left = int(x1 * image_width)
            top = int(y1 * image_height)
            right = int(x2 * image_width)
            bottom = int(y2 * image_height)
            bounds_list.append([left, top, right, bottom])

    # 可视化结果
    if visualize:
        print(f"Detected {len(bounds_list)} UI elements")
        visualize_detection_results(screenshot_path, bounds_list)
        visualize_parsed_details(parsed_content_list)
    
    return bounds_list, parsed_content_list

def find_clicked_element(bounds_list, click_x, click_y):
    """找到包含点击位置的最小边界框"""
    smallest_bounds = None
    smallest_area = float('inf')

    for bounds in bounds_list:
        left, top, right, bottom = bounds
        # 检查点击位置是否在边界框内
        if left <= click_x <= right and top <= click_y <= bottom:
            area = (right - left) * (bottom - top)
            if area < smallest_area:
                smallest_area = area
                smallest_bounds = bounds

    return smallest_bounds
# 完整的测试函数
def test_click_detection(screenshot_path, click_x, click_y):
    """完整的测试流程：检测元素并找到点击位置"""
    
    # 提取边界框
    bounds_list, parsed_content_list = extract_all_bounds_with_visualization(screenshot_path, visualize=True)
    
    # 找到点击的元素
    clicked_bounds = find_clicked_element(bounds_list, click_x, click_y)
    
    # 可视化点击结果
    if clicked_bounds:
        print(f"\n✅ Clicked element found at bounds: {clicked_bounds}")
        visualize_detection_results(
            screenshot_path, 
            bounds_list, 
            clicked_bounds=clicked_bounds,
            click_point=(click_x, click_y)
        )
        
        # 保存结果图片
        output_path = screenshot_path.replace('.png', '_detection.png').replace('.jpg', '_detection.jpg')
        save_detection_image(screenshot_path, bounds_list, output_path, clicked_bounds)
    else:
        print(f"\n❌ No element found at click position ({click_x}, {click_y})")
        visualize_detection_results(
            screenshot_path, 
            bounds_list, 
            click_point=(click_x, click_y)
        )
    
    return clicked_bounds

# 使用示例
if __name__ == "__main__":
    # 测试你的代码
    screenshot_path = "/Users/fengyunfei/Desktop/mobiagent/Mobibench/data/美团/type1/4/5.jpg"  # 替换为你的截图路径
    click_x, click_y = 100, 200  # 替换为你的点击坐标
    
    # 运行测试
    clicked_element = test_click_detection(screenshot_path, click_x, click_y)