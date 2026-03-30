import uiautomator2 as u2
import time
import os
import shutil
import base64
from PIL import Image
import io
import json
import re
import logging
import sys
import json
from openai import OpenAI
import argparse
from pathlib import Path
from MobiBench.utils.draw_bounds import process_folder
from MobiBench.utils.task_get import get_tasks,get_tasks_1
from MobiBench.env.fsm import build_AppFSM, quick_build_AppFSM
from datetime import datetime
from MobiBench.env.type_spaces import Action
AUTO_DECIDER = '''
## 角色定义
你是一个手机操作AI助手，需要帮助用户完成指定任务。

## 输入说明
我会提供给你：
1. **操作历史**：之前的所有操作记录
2. **屏幕截图**：当前手机屏幕的完整截图
3. **{layer_count}张标注截图**：基于屏幕截图生成的可点击元素标注图层。为避免元素重叠，将所有可点击元素分配到不同图层中显示，所有图层包含的元素合集即为全部可点击元素
   - 可点击元素用红色方框标出
   - 每个元素的编号(index)显示在红色方框的左上角内侧，红底白字数字

## 可用操作
1. **点击操作 (click)**
    - 参数：index (整数，对应标注截图中要点击的UI元素编号)
    - 参数：target_element (字符串，描述要点击的UI元素)
    - **重要**：必须仔细观察标注截图，找到与target_element描述最匹配的红色方框，使用该方框左上角内侧红底白字显示的数字作为index
    - **防止误选**：在reasoning中必须明确说明为什么选择这个红色方框而不是其相邻的红色方框
   
2. **滑动操作 (swipe)**
    - 参数：direction (字符串，必须是 UP、DOWN、LEFT、RIGHT 之一)
    - **重要**：滑动方向说明：UP表示向上滑动手指来向上滚动内容并显示下方内容；DOWN表示向下滑动手指来向下滚动内容并显示上方内容；LEFT表示向左滑动手指来向左滚动内容；RIGHT表示向右滑动手指来向右滚动内容。
   
3. **文本输入 (input)**
    - 参数：text (字符串，要输入的文本内容)

4. **回退 (back)**
    - 无参数，表示回退到上一状态
   
5. **完成任务 (done)**
    - 无参数，表示任务已完成

## 输出格式
请严格按照以下JSON格式输出：
```json
{{
    "reasoning": "详细说明你的分析思路和选择这个操作的原因",
    "action": "操作名称(click/swipe/input/done)",
    "parameters": {{
        "参数名": "参数值"
  }}
}}
```

## index选择的关键步骤（点击操作必读）
**步骤1: 精确描述目标元素的视觉特征**
- 详细描述目标元素的内容、颜色、形状等视觉特征
- 精确描述元素在屏幕中的位置（如：屏幕上方1/3处、左侧边缘、右下角等）
- 描述元素周围的其他UI元素作为参考点
- 例如："需要点击带有'搜索'文字的白色输入框，位于屏幕最上方，在应用标题下方"

**步骤2: 系统性查找红色方框**
- **必须按顺序逐张查看每一张标注图**，不能跳过任何一张
- 对于每张图，先整体观察所有红色方框的分布
- 重点查找与步骤1描述的位置和特征完全匹配的红色方框
- **关键要求：红色方框必须完全包围目标元素，边界贴合**

**步骤3: 多重验证确保选择正确（最重要步骤）**
- **位置验证**：确认红色方框的位置与步骤1描述的位置完全一致
- **内容验证**：仔细观察红色方框内部包含的内容是否就是目标元素
- **边界验证**：红色方框的边界应该紧贴目标元素，不应该包含过多空白区域
- **排除干扰**：如果有多个相似的红色方框，必须选择位置最精确匹配的那个
- **避免相邻选择**：绝对不能选择目标元素旁边或附近的红色方框

**步骤4: 读取index数字（执行前的最后确认）**
- 再次确认选中的红色方框确实包围了正确的目标元素
- 查看该红色方框**左上角内侧**的红底白字数字
- **严格要求：必须是左上角，数字必须清晰可见**
- 该数字就是要使用的index值

**步骤5: 最终验证**
- 在reasoning中明确说明："我选择的红色方框位于[具体位置]，框内包含[具体内容]，左上角数字为[X]"
- 如果对选择有任何不确定，必须重新从步骤1开始

## 文本输入的关键步骤（输入操作必读）
**重要前提：绝对禁止在未激活输入框时直接使用input操作！**

**步骤1: 强制检查软键盘状态（必须执行，不可跳过）**
- **必须检查**：仔细观察当前屏幕最底部是否显示了软键盘（虚拟键盘界面）
- **判断标准**：如果屏幕底部没有显示包含字母、数字键的软键盘界面，说明没有任何输入框被激活
- **关键规则**：**只有当软键盘完全显示在屏幕底部时，才允许进行input操作**
- **reasoning必须写明**："检查软键盘状态：[已显示/未显示]"

**步骤2: 输入框激活操作（如果第1步检查失败则必须执行）**
- **严格禁止**：如果没有软键盘或输入框未激活，绝对不能使用input操作
- **必须操作**：必须先使用click操作点击目标输入框来激活它
- **reasoning必须写明**："软键盘未显示/输入框未激活，必须先点击激活输入框"

**步骤3: 处理现有内容**
- 如果输入框中有默认文本，可以先尝试清除或直接覆盖
- 根据具体情况选择处理方式

**步骤4: 执行文本输入（仅在前置条件满足时）**
- 确认软键盘已显示且输入框已激活后，才能使用input操作
- 输入后检查输入框内的文本是否正确，确保没有输入错误、遗漏或多余字符
- 在reasoning中必须明确说明"已确认软键盘显示且输入框已激活"

**步骤5: 输入后的软键盘处理（重要）**
- **输入完成后必须检查**：观察软键盘上的按键类型
- **隐藏软键盘的判断标准**：
  - 如果软键盘上有"搜索"、"确定"、"完成"、"发送"等提交按钮，应该点击这些按钮
  - 如果软键盘上只有"下一项"、"换行"等非提交按钮，且软键盘遮挡了重要的界面元素，应该点击软键盘右上角的"向下箭头"按钮来隐藏软键盘
- **reasoning必须说明**："检查软键盘按键类型：[提交类型/导航类型]。软键盘是否遮挡重要元素：[是/否]。决定[点击提交按钮/隐藏软键盘/保持现状]"

**严格禁止的input操作模式**
1. 没有软键盘但直接input
2. 未在reasoning中说明检查过程但直接input
3. 看到输入框就直接input（必须先检查激活状态）
4. 输入完成后不考虑软键盘遮挡问题

**唯一正确的input操作模式**
- reasoning包含："检查软键盘状态：已显示。检查输入框状态：已激活。已确认可以进行文本输入。"
- 只有包含以上完整检查过程的reasoning才允许使用input操作
- **输入后处理**：输入完成后，必须检查软键盘按键类型和是否遮挡重要元素，决定是否需要隐藏软键盘

## 重要规则
1. **位置匹配优先**：先确定元素在原图中的准确位置，再找标注图中对应位置的红色方框
2. **数字读取准确**：index必须是红色方框左上角内侧红底白字显示的实际数字
3. **避免误选相邻元素**：这是最容易出错的地方！必须确保选择的红色方框完全包围目标元素，而不是相邻的类似元素
4. **强制性相邻元素排除检查**：在选择任何index前，必须明确说明为什么没有选择周围的其他红色方框
5. **软键盘遮挡处理**：输入完成后，如果软键盘遮挡了重要元素且没有提交按钮，应该点击右上角向下箭头隐藏软键盘
6. **多步骤操作**：对于复杂选择（如日期范围、时间段、级联选项），需要多个连续操作
7. **日期选择特别注意**：
   - 在日期选择界面时，必须先确认当前显示的月份是否正确
   - 不能仅仅看到相同的日期数字就直接选择，必须确保月份匹配任务要求
   - 如果月份不对，需要先切换到正确的月份，然后再选择日期
8. **任务完成判断**：只有在确实完成了指定任务时才使用done操作
9. **操作连贯性**：每个操作都应该基于当前屏幕状态和任务目标进行合理选择
10. **页面错误处理**：如果遇到进入错误页面或加载失败，可以尝试返回上一级界面（通过手势自屏幕最左侧向右滑动或使用点击返回按钮）

## index选择示例
**错误示例1**：
- reasoning: "需要点击搜索按钮"
- 问题：没有描述元素的具体位置和视觉特征

**错误示例2**：
- reasoning: "需要点击搜索框，位于屏幕上方。在标注图中找到了搜索框，选择数字8。"
- 问题：描述过于简单，没有验证过程，容易选错相邻元素

**正确示例**：
- reasoning: "1）目标元素详细描述：需要点击带有'搜索'提示文字的白色输入框，该输入框呈长方形，有浅灰色边框。2）精确位置描述：该搜索框位于屏幕最上方，在状态栏下方约50像素处，占据屏幕宽度的80%左右，位置居中。3）标注图查找：在第2张标注图中，我找到了位于屏幕上方中央位置的红色方框。4）红色方框验证：该红色方框完全包围了搜索输入框，边界与输入框的边缘完全贴合，框内确实包含带有'搜索'文字的白色输入框。5）index读取：该红色方框的左上角内侧清晰显示数字'15'。6）最终确认：确认该方框没有包含其他无关元素，也不是相邻的其他UI元素，正是我要点击的搜索框。"
- parameters: {{"index": 15, "target_element": "搜索输入框"}}


## 你的任务为：
{task_description}

## 操作历史
{history}

**记住：每次点击操作都必须在reasoning中包含完整的6步验证过程，确保精确匹配而不是选择相邻元素！每次输入操作后都要考虑软键盘遮挡问题！**


'''


decider_prompt_template = AUTO_DECIDER.replace("````markdown", "").replace("````", "").strip()
device = None  # 设备连接对象
hierarchy = None  # 层次结构数据
data_index = 1  # 数据索引

operation_history = []  # 操作历史记录
logger = None  # 日志记录器

# 全局配置变量，将由命令行参数设置
max_steps = 15



# 将路径 img_path 截图保存为base64编码的字符串
def get_screenshot(img_path, factor=0.4):
    img = Image.open(img_path)
    #img = img.resize((int(img.width * factor), int(img.height * factor)), Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    screenshot = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return screenshot


# ==== NEW: 一个简单的名字清洗函数，用于生成安全的目录名 ====
def _safe_name(text: str, max_len: int = 50) -> str:
    safe = "".join(c if c.isalnum() or c in ("_", "-", " ") else "_" for c in text)
    safe = "_".join(safe.split())  # 把空格压缩成单个下划线
    return safe[:max_len] or "task"



def get_std_act(act,parameters,action_history,reasoning_history,reasoning,bounds_list):

    if act in ["done","完成"]:
        logger.info("任务完成！")
        action = {
            "reasoning": reasoning,
            "function": {
                "name": "done",
                "parameters": {}
            }
        }
        logger.info(f"完成操作: {action}")
        action_history.append(action)
        reasoning_history.append(reasoning)
        return Action(act_type='done',parameters={})
    
    elif act.lower() == "click":
        target_element = parameters.get("target_element")
        index = parameters.get("index")
        if index is None or index < 0 or index >= len(bounds_list):
            logger.error(f"错误：index {index} 超出范围，有效范围为 0 到 {len(bounds_list)-1}")
        
        # 
        if index >= len(bounds_list):
            return None
        
        bounds = bounds_list[index]
        # index, bounds = decide_click_element(data_dir, action_count + 1, task_description, reasoning, target_element)
        logger.info(f"选择点击元素: {target_element} (index: {index}, bounds: {bounds})")
        x = (bounds[0] + bounds[2]) / 2
        y = (bounds[1] + bounds[3]) / 2
        #handle_click(x, y)
        action = {
            "reasoning": reasoning,
                "function": {
                "name": "click",
                "parameters": {
                    "position_x": x,
                    "position_y": y,
                    "bounding_box": bounds,
                    "target_element": target_element,
                }
            }
        }
        logger.info(f"点击操作: {action}")
        action_history.append(action)
        reasoning_history.append(reasoning)
        return Action(act_type="click", parameters={"position_x": x, "position_y": y})

    elif act.lower() == "input":
        text = parameters.get("text")
        #handle_input(text)
        action = {
            "reasoning": reasoning,
            "function": {
                "name": "input",
                "parameters": {
                    "text": text
                }
            }
        }
        logger.info(f"输入操作: {action}")
        action_history.append(action)
        reasoning_history.append(reasoning)
        return Action(act_type="input", parameters={"text": text})



    elif act.lower() == "swipe":
        direction = parameters.get("direction").lower()
        #handle_swipe(direction)
        action = {
            "reasoning": reasoning,
            "function": {
                "name": "swipe",
                "parameters": {
                    "direction": direction
                }
            }
        }
        logger.info(f"滑动操作: {action}")
        action_history.append(action)
        reasoning_history.append(reasoning)
        return Action(act_type="swipe", parameters={"direction":direction})
    
    elif act.lower() == "back":
        action = {
            "reasoning": reasoning,
            "function": {
                "name": "back",
                "parameters": {}
            }
        }
        logger.info(f"滑动操作: {action}")
        action_history.append(action)
        reasoning_history.append(reasoning)
        return Action(act_type="back", parameters={})
    
    else:
        return None


def run(
    fsm,
    args,
    app: str,
    task: str,
    instruction: str,
    runs_dir: str,
    client,
    model
):
    logger.info(f"开始执行任务: {app} | {task} |{instruction}")
    action_history = []
    reasoning_history = []
    screenshots = []
    max_steps = fsm.max_op_times

    
    pre_state = None
    layer_count, bounds_list = 0,[]
    layer_folder_path = None
    for step in range(1,max_steps,1):
        
        cur = fsm.cur_state
        
        img_path = cur.img_path
        logger.info("==== Step %d | State: %s (%s) ====", step, img_path, cur.cluster_class)
        
        action_count = len(action_history)  # 已有的操作数量
        action_index = action_count + 1     # 接下来的操作索引

        

        if action_count == 0:
            history = "(No history)"
        else:
            history = "\n".join(f"{idx}. {h}" for idx, h in enumerate(reasoning_history, 1))
        
        if pre_state == None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_inst = _safe_name(instruction)
            run_dir = os.path.join(runs_dir, app, task, f"{timestamp}_{safe_inst}")
            os.makedirs(run_dir, exist_ok=True)
            layer_folder_path = os.path.join(run_dir,img_path[-10:])
            os.makedirs(layer_folder_path , exist_ok=True)

            layer_count, bounds_list = process_folder(img_path,layer_folder_path)
        else:
            if pre_state.img_path != img_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_inst = _safe_name(instruction)
                run_dir = os.path.join(runs_dir, app, task, f"{timestamp}_{safe_inst}")
                os.makedirs(run_dir, exist_ok=True)
                layer_folder_path = os.path.join(run_dir,img_path[-10:])
                os.makedirs(layer_folder_path , exist_ok=True)
                layer_count, bounds_list = process_folder(img_path,layer_folder_path)
            else:
                pass 

        if layer_count == 0:
            logger.info(f"处理 {layer_folder_path} 失败，跳过此步骤")
            continue
        logger.info(f"已处理 {layer_folder_path}，共绘制 {layer_count} 个图层")

        # decider_prompt - 使用最终的任务描述
        decider_prompt = decider_prompt_template.format(
            task_description = instruction,  # 使用改写后的任务描述
            history = history,
            layer_count = layer_count
        )
        message_content = [
            {"type": "text", "text": decider_prompt}
        ]
        message_content.append({
            "type": "text",
            "text": f"\n屏幕截图:"
        })
        cur_screenshot = get_screenshot(img_path)
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{cur_screenshot}"}
        })
        # 遍历所有标注图层
        for idx in range(1, layer_count + 1):
            screenshot_path = os.path.join(layer_folder_path, f"layer_{idx}.jpg")
            screenshot = get_screenshot(screenshot_path)
            message_content.append({
                "type": "text", 
                "text": f"\n第{idx}张标注图层:"
            })
            message_content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"}
            })
        decider_response_str = client.chat.completions.create(
            model= model,
            messages=[
                {
                    "role": "user",
                    "content": message_content
                }
            ]
        ).choices[0].message.content
        
        logger.info(f"response: \n{decider_response_str}")
        pattern = re.compile(r"```json\n(.*)\n```", re.DOTALL)
        match = pattern.search(decider_response_str)
        if not match:
            logger.error("错误：未找到有效的JSON响应")
            continue
        decider_response = json.loads(match.group(1))

        reasoning = decider_response.get("reasoning")
        action = decider_response.get("action")
        parameters = decider_response.get("parameters")

        std_act = get_std_act(act=action,parameters=parameters,action_history=action_history,reasoning=reasoning,reasoning_history=reasoning_history,bounds_list=bounds_list)
        pre_state = fsm.cur_state
        if std_act!=None:
            if std_act.act_type == "done":
                logger.info("模型输出 finished(...)，结束交互。")
                
                break
            fsm.action(std_act)
        else:
            fsm.is_failed = True
        new_state = fsm.cur_state
        
        if new_state.cluster_class in ("DONE", "Done", "done"):
            logger.info("到达 DONE 状态，任务完成！")
            done_flag = True
            break
        if fsm.is_failed:
            logger.info("到达 FAILED 状态，任务失败！")
            break


        

  


    
    

def setup_logger(data_dir):
    """设置日志记录器，同时输出到控制台和文件"""
    global logger
    
    # 创建日志目录
    log_file = os.path.join(data_dir, "execution.log")
    
    # 创建logger，使用特定名称避免冲突
    logger_name = f'auto_collect_{id(data_dir)}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 防止日志传播到根logger
    logger.propagate = False
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def change_auto_data(data_log_path, index):
    parse_error = os.path.join(data_log_path, "parse.error")
    if os.path.exists(parse_error):
        return
    task_data = os.path.join(data_log_path, "task_data.json")
    if not os.path.exists(task_data):
        return

    with open(task_data, 'r', encoding='utf-8') as file:
        task_data = json.load(file)

    app_name = task_data.get("app_name")
    task_type = None
    task_description = task_data.get("task_description")
    actions = task_data.get("actions")

    new_actions = []
    for action in actions:
        action_type = action["function"]["name"].lower()
        if action_type == "click":
            new_action = {
                "type": action_type,
                "position_x": int(action["function"]["parameters"]["position_x"]),
                "position_y": int(action["function"]["parameters"]["position_y"]),
                "bounds": action["function"]["parameters"]["bounding_box"]
            }
            new_actions.append(new_action)
        elif action_type == "swipe":
            new_action = {
                "type": action_type,
                "press_position_x": None,
                "press_position_y": None,
                "release_position_x": None,
                "release_position_y": None,
                "direction": action["function"]["parameters"]["direction"]
            }
            new_actions.append(new_action)
        elif action_type == "input":
            new_action = {
                "type": action_type,
                "text": action["function"]["parameters"]["text"]
            }
            new_actions.append(new_action)
        elif action_type == "done":
            new_action = {
                "type": "done"
            }
            new_actions.append(new_action)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    data = {
        "app_name": app_name,
        "task_type": task_type,
        "task_description": task_description,
        "action_count": len(new_actions),
        "actions": new_actions
    }

    dest_path_dir = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(dest_path_dir):
        os.makedirs(dest_path_dir)
    existing_dirs = [d for d in os.listdir(dest_path_dir) if os.path.isdir(os.path.join(dest_path_dir, d)) and d.isdigit()]
    if existing_dirs:
        max_index = max(int(d) for d in existing_dirs) + 1
    else:
        max_index = 1
    dest_path = os.path.join(dest_path_dir, str(max_index))
    os.makedirs(dest_path)
    
    # 复制并重命名图片文件
    for index in range(1, len(new_actions) + 2):  # +2 因为通常有一张额外的截图
        screenshot_src = os.path.join(data_log_path, str(index), "screenshot.jpg")
        if os.path.exists(screenshot_src):
            screenshot_dest = os.path.join(dest_path, f"{index}.jpg")
            shutil.copy2(screenshot_src, screenshot_dest)
            print(f"复制图片: {screenshot_src} -> {screenshot_dest}")
        
    with open(os.path.join(dest_path, "actions.json"), 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Auto collection of GUI data')
    parser.add_argument('--model', type=str,  help='name of the LLM model')
    parser.add_argument('--api_key', type=str,  help='API key for the LLM model')
    parser.add_argument('--base_url', type=str, help='base URL for the LLM model API')
    parser.add_argument('--max_steps', type=int, default=15, help='maximum steps per task (default: 15)')
    parser.add_argument("--data_root", default="/Users/fff/Desktop/mobiagent/MobiBench/data", help="MobiBench data 根目录（包含 rawdata/）")
    parser.add_argument(
        "--runs_dir",
        default=r"/Users/fff/Desktop/mobiagent/MobiBench/agents/gemini/layers1",  # ==== NEW: 所有运行结果的根目录 ====
        help="所有运行结果的根目录，用于保存轨迹和坐标",
    )
    parser.add_argument("--task_json", default="/Users/fff/Desktop/mobiagent/MobiBench/data/test.json", help="task json file")
    parser.add_argument("--result_dir", default="/Users/fff/Desktop/mobiagent/MobiBench/results/dev", help="result directory")
    parser.add_argument("--log_dir", default="/Users/fff/Desktop/mobiagent/MobiBench/agents/gemini/log1", help="log directory")
    args = parser.parse_args()
    
    # 设置全局配置
    model = args.model
    api_key = args.api_key
    base_url = args.base_url
    max_steps = args.max_steps
    
    # 初始化OpenAI客户端
    client = OpenAI(
            api_key= args.api_key,
            base_url=args.base_url
        )
    with open(args.task_json, 'r', encoding='utf-8') as f:
        alldata = json.load(f)
    datapath = args.data_root
    data_log_dir = args.log_dir
    for app in alldata.keys():
        for tasktype in alldata[app]:
            tasklist = get_tasks(app, tasktype)
            #logger.info("构建 FSM 中…")
            fsm = build_AppFSM(app=app, task=tasktype, data_path=datapath)
            # 让 FSM 内部的 max_op_times 和 CLI 一致
            fsm.max_op_times = args.max_steps

            for task in tasklist:
                print(f"任务: {task}，应用: {app}，类型: {tasktype}")
                logger = setup_logger(data_log_dir)
                logger.info("程序启动")
                
                fsm._reset()
                
                start = time.time()
                run(
                    fsm=fsm,
                    args=args,
                    app=app,
                    task=tasktype,
                    instruction=task,
                    runs_dir=args.runs_dir,  # ==== NEW: 传入 runs 根目录 ====
                    client=client,
                    model="gemini-2.5-flash"

                )
                end = time.time()
                from MobiBench.utils.score_proc import save_result
                save_result(
                    md="gemini-2.5-flash",
                    app=app,
                    task=tasktype,
                    inst=task,
                    fsm=fsm,
                    time_use=end-start,
                    savepath=args.result_dir,
                )
                from MobiBench.utils.score_proc import save_visited_result
                save_visited_result(
                    md="gemini-2.5-flash",
                    app=app,
                    task=tasktype,
                    fsm=fsm,
                    savepath=args.result_dir + "/visited",
                )
    
    
