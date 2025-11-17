import os
import base64
import json
import glob
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI


class OpenAIVisualTagger:
    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash"):
        """初始化 OpenAI 客户端"""
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")
        
        self.client = OpenAI(api_key=api_key,
                             base_url=f"http://ipads.chat.gpt:3006/v1")
        self.model = model
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """将图片编码为 base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def prepare_image_message(self, image_path: str, description: str = None) -> List[Dict]:
        """准备图片消息"""
        base64_image = self.encode_image_to_base64(image_path)
        
        messages = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                }
            }
        ]
        
        if description:
            messages.append({
                "type": "text",
                "text": description
            })
            
        return messages
    
    def analyze_single_screenshot(self, 
                                image_path: str, 
                                mark_description: str,
                                positive_example_images: List[str],
                                negative_example_images: List[str]) -> Dict[str, Any]:
        """
        分析单张截图是否需要打标记
        
        Args:
            image_path: 待分析的截图路径
            mark_description: 标记的文本描述
            positive_example_images: 正例图片路径列表（需要打标记的示例）
            negative_example_images: 反例图片路径列表（不需要打标记的示例）
        """
        
        try:
            # 构建消息内容
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self._build_system_prompt(mark_description)
                        }
                    ]
                }
            ]
            
            # 构建用户消息
            user_content = []
            
            # 添加正例图片
            user_content.append({
                "type": "text",
                "text": "以下是一些需要打标记的示例图片（正例）："
            })
            
            for i, example_path in enumerate(positive_example_images[:3]):  # 限制数量避免token超限
                if os.path.exists(example_path):
                    user_content.extend(self.prepare_image_message(
                        example_path
                       
                    ))
            
            # 添加反例图片
            user_content.append({
                "type": "text", 
                "text": "以下是一些不需要打标记的示例图片（反例）："
            })
            
            for i, example_path in enumerate(negative_example_images[:3]):
                if os.path.exists(example_path):
                    user_content.extend(self.prepare_image_message(
                        example_path
                       
                    ))
            
            # 添加待分析的图片
            user_content.append({
                "type": "text",
                "text": "请分析以下截图，基于上面提供的正例和反例图片，判断是否需要打标记："
            })
            user_content.extend(self.prepare_image_message(
                image_path,
                f"待分析的截图: {os.path.basename(image_path)}"
            ))
            
            messages.append({
                "role": "user",
                "content": user_content
            })
            
            # 调用 OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,

                temperature=0.0
            )
            
            # 解析响应
            print(response)
            result = self._parse_response(response.choices[0].message.content)

            result['image_path'] = image_path
            result['success'] = True
            
            return result
            
        except Exception as e:
            return {
                'image_path': image_path,
                'success': False,
                'error': str(e),
                'should_mark': False,
                'confidence': 0.0,
                'reasoning': ''
            }
    
    def _build_system_prompt(self, mark_description: str) -> str:
        """构建系统提示词"""
        
        return f"""
            你是一个专业的视觉分析助手。你的任务是通过对比示例图片，分析给定的截图是否需要打上特定标记。

            **标记描述**：{mark_description}

            **学习方式**：
            - 研究用户提供的正例图片（需要打标记的示例）和反例图片（不需要打标记的示例）
            - 学习这些示例图片中的视觉模式、界面特征、颜色、布局等
            - 基于学到的模式以及标记描述来判断新图片是否需要打标记

            **输出格式**：
            请严格按照以下 JSON 格式返回结果，不要包含其他任何文本：
            {{
                "should_mark": true/false,
                "confidence": 0.0-1.0,
                "reasoning": "详细的分析理由，包括：1) 与正例图片的相似之处 2) 与反例图片的不同之处 3) 具体的视觉特征分析"
            }}

            **分析重点**：
            1. 界面布局和结构
            2. 颜色 scheme 和视觉样式
            3. 图标、按钮和控件的类型
            4. 文字内容和语言特征
            5. 整体视觉氛围和风格

            **置信度说明**：
            - 0.9-1.0: 与正例图片高度相似
            - 0.7-0.9: 有明显正例特征
            - 0.5-0.7: 有一些正例特征
            - 0.3-0.5: 特征不明显
            - 0.0-0.3: 与反例图片更相似

            请基于视觉对比给出准确判断！
            """
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """解析模型响应"""
        try:
            response_text = response_text.strip()
            
            # 提取 JSON 部分
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                raise ValueError("未找到有效的 JSON 格式")
            
            json_str = response_text[start_idx:end_idx+1]
            result = json.loads(json_str)
            
            # 验证必需字段
            required_fields = ['should_mark', 'confidence', 'reasoning']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"缺少必需字段: {field}")
            
            # 确保数据类型正确
            result['should_mark'] = bool(result['should_mark'])
            result['confidence'] = float(result['confidence'])
            
            return result
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"解析响应失败: {e}")
            return {
                'should_mark': False,
                'confidence': 0.0,
                'reasoning': f'响应解析失败: {str(e)}'
            }
    
    def batch_analyze_screenshots(self, 
                                screenshot_dir: str,
                                mark_description: str,
                                positive_example_images: List[str],
                                negative_example_images: List[str],
                                image_extensions: List[str] = None) -> List[Dict[str, Any]]:
        """
        批量分析截图目录中的所有图片
        """
        if image_extensions is None:
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        
        # 获取所有图片文件
        image_files = []
        for ext in image_extensions:
            pattern = os.path.join(screenshot_dir, f"**/{ext}")
            image_files.extend(glob.glob(pattern, recursive=True))
        
        print(f"找到 {len(image_files)} 张截图")
        print(f"使用 {len(positive_example_images)} 张正例图片")
        print(f"使用 {len(negative_example_images)} 张反例图片")
        print("开始分析...")
        
        results = []
        for i, image_path in enumerate(image_files, 1):
            print(f"分析进度: {i}/{len(image_files)} - {os.path.basename(image_path)}")
            
            result = self.analyze_single_screenshot(
                image_path=image_path,
                mark_description=mark_description,
                positive_example_images=positive_example_images,
                negative_example_images=negative_example_images
            )
            results.append(result)
            
            # 添加延迟避免 API 限制
            import time
            time.sleep(1)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """保存分析结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"分析结果已保存到: {output_file}")
        self.print_statistics(results)
    
    def print_statistics(self, results: List[Dict[str, Any]]):
        """打印统计信息"""
        successful = [r for r in results if r.get('success', False)]
        marked = [r for r in successful if r.get('should_mark', False)]
        
        print(f"\n=== 分析统计 ===")
        print(f"总图片数: {len(results)}")
        print(f"成功分析: {len(successful)}")
        print(f"需要标记: {len(marked)}")
        
        if successful:
            avg_confidence = sum(r.get('confidence', 0) for r in successful) / len(successful)
            print(f"平均置信度: {avg_confidence:.2f}")
            
            # 高置信度标记的图片
            high_confidence_marked = [r for r in marked if r.get('confidence', 0) > 0.8]
            if high_confidence_marked:
                print(f"\n高置信度标记的图片 (置信度 > 0.8):")
                for result in high_confidence_marked[:5]:  # 只显示前5个
                    print(f"- {os.path.basename(result['image_path'])} (置信度: {result['confidence']:.2f})")

# 使用示例
def main():
    # 初始化标记器
    tagger = OpenAIVisualTagger(api_key="sk-rfCIGhxrzcdsMV4jC17e406bE56c47CbA5416068A62318D3")
    
    # 标记描述
    MARK_DESCRIPTION = "包含错误提示或异常状态的界面"
    
    # 自动查找示例图片
    def find_example_images(example_dir: str) -> List[str]:
        """查找示例目录中的所有图片"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_files = []
        for ext in extensions:
            pattern = os.path.join(example_dir, f"**/{ext}")
            image_files.extend(glob.glob(pattern, recursive=True))
        return image_files
    
    # 查找正例和反例图片
    POSITIVE_EXAMPLE_IMAGES = find_example_images("./examples/positive")
    NEGATIVE_EXAMPLE_IMAGES = find_example_images("./examples/negative")
    
    if not POSITIVE_EXAMPLE_IMAGES:
        print("错误: 在 ./examples/positive 目录中未找到正例图片")
        return
        
    if not NEGATIVE_EXAMPLE_IMAGES:
        print("错误: 在 ./examples/negative 目录中未找到反例图片")
        return
    
    print(f"找到 {len(POSITIVE_EXAMPLE_IMAGES)} 张正例图片")
    print(f"找到 {len(NEGATIVE_EXAMPLE_IMAGES)} 张反例图片")
    
    # 批量分析截图
    results = tagger.batch_analyze_screenshots(
        screenshot_dir="./examples/target",
        mark_description=MARK_DESCRIPTION,
        positive_example_images=POSITIVE_EXAMPLE_IMAGES,
        negative_example_images=NEGATIVE_EXAMPLE_IMAGES
    )
    
    # 保存结果
    tagger.save_results(results, "visual_analysis_results.json")

if __name__ == "__main__":
    main()