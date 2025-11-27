# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import re
import ast
import math
import json
from typing import Any, Dict, List, Tuple

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def convert_point_to_coordinates(text, is_answer=False):
    # 匹配 <bbox> 后面的四个数字
    pattern = r"<point>(\d+)\s+(\d+)</point>"

    def replace_match(match):
        x1, y1 = map(int, match.groups())
        x = (x1 + x1) // 2  # 使用截断取整
        y = (y1 + y1) // 2  # 使用截断取整
        if is_answer:
            return f"({x},{y})"  # 只返回 (x, y) 格式
        return f"({x},{y})"  # 返回带标签的格式

    # 去掉 [EOS] 并替换 <bbox> 坐标
    text = re.sub(r"\[EOS\]", "", text)
    return re.sub(pattern, replace_match, text).strip()


# 定义一个函数来解析每个 action
def parse_action(action_str):
    try:
        # 解析字符串为 AST 节点
        node = ast.parse(action_str, mode='eval')

        # 确保节点是一个表达式
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")

        # 获取表达式的主体
        call = node.body

        # 确保主体是一个函数调用
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        # 获取函数名
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        # 获取关键字参数
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            # 处理不同类型的值，这里假设都是常量
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # 兼容旧版本 Python
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {'function': func_name, 'args': kwargs}

    except Exception as e:
        print(f"Failed to parse action '{action_str}': {e}")
        return None


def escape_single_quotes(text):
    # 匹配未转义的单引号（不匹配 \\'）
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)


def round_by_factor(number: float, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return int(round(number / factor) * factor)


def ceil_by_factor(number: float, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return int(math.ceil(number / factor) * factor)


def floor_by_factor(number: float, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return int(math.floor(number / factor) * factor)


def linear_resize(height: int,
                  width: int,
                  factor: int = IMAGE_FACTOR,
                  min_pixels: int = MIN_PIXELS,
                  max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    if width * height > max_pixels:
        """
        如果图片超过/低于像素限制，则计算一个缩放因子resize_factor，使图片的像素数缩小到等于或小于max_pixels。这个缩放因子是通过开平方根计算的，确保纵横比保持不变,这样原始的相对坐标可以不经转换直接复用
        """
        resize_factor = math.sqrt(max_pixels / (width * height))
        width, height = int(width * resize_factor), int(height * resize_factor)
    if width * height < min_pixels:
        resize_factor = math.sqrt(min_pixels / (width * height))
        width, height = math.ceil(width * resize_factor), math.ceil(
            height * resize_factor)

    return height, width


def smart_resize(height: int,
                 width: int,
                 factor: int = IMAGE_FACTOR,
                 min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def parse_action_to_structure_output(text,
                                     factor,
                                     origin_resized_height,
                                     origin_resized_width,
                                     model_type="qwen25vl",
                                     max_pixels=16384 * 28 * 28,
                                     min_pixels=100 * 28 * 28):
    text = text.strip()

    # initialize smart resize values to the original resized dimensions so they are always defined
    smart_resize_height, smart_resize_width = origin_resized_height, origin_resized_width

    if "<point>" in text:
        text = convert_point_to_coordinates(text)
    if "start_point=" in text:
        text = text.replace("start_point=", "start_box=")
    if "end_point=" in text:
        text = text.replace("end_point=", "end_box=")
    if "point=" in text:
        text = text.replace("point=", "start_box=")

    if model_type == "qwen25vl":
        smart_resize_height, smart_resize_width = smart_resize(
            origin_resized_height,
            origin_resized_width,
            factor=IMAGE_FACTOR,
            min_pixels=min_pixels,
            max_pixels=max_pixels)

    # 正则表达式匹配 Action 字符串
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action: |$)"
        thought_hint = "Thought: "
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action: |$)"
        thought_hint = "Reflection: "
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action: |$)"
        thought_hint = "Action_Summary: "
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action: |$)"
        thought_hint = "Thought: "
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
    assert "Action" in text
    action_str = text.split("Action: ")[-1]

    tmp_all_action = action_str.split(")\n\n")
    all_action = []
    for action_str in tmp_all_action:
        if "type(content" in action_str:
            if not action_str.strip().endswith(")"):
                action_str = action_str.strip() + ")"
            # 正则表达式匹配 content 中的字符串并转义单引号
            def escape_quotes(match):
                content = match.group(1)  # 获取 content 的值
                return content

            # 使用正则表达式进行替换
            pattern = r"type\(content='(.*?)'\)"  # 匹配 type(content='...')
            if re.search(pattern, action_str):  # 检查是否有匹配项
                content = re.sub(pattern, escape_quotes, action_str)
            else:
                raise ValueError("Pattern not found in the input string.")

            # 处理字符串
            action_str = escape_single_quotes(content)
            action_str = "type(content='" + action_str + "')"
        if not action_str.strip().endswith(")"):
            action_str = action_str.strip() + ")"
        all_action.append(action_str)

    parsed_actions = [
        parse_action(action.replace("\n", "\\n").lstrip())
        for action in all_action
    ]
    actions = []
    for action_instance, raw_str in zip(parsed_actions, all_action):
        if action_instance == None:
            print(f"Action can't parse: {raw_str}")
            raise ValueError(f"Action can't parse: {raw_str}")
        action_type = action_instance["function"]
        params = action_instance["args"]

        # import pdb; pdb.set_trace()
        action_inputs = {}
        for param_name, param in params.items():
            if param == "": continue
            param = param.lstrip()  # 去掉引号和多余的空格
            # 处理start_box或者end_box参数格式 '<bbox>x1 y1 x2 y2</bbox>'
            action_inputs[param_name.strip()] = param

            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                # Remove parentheses and split the string by commas
                numbers = ori_box.replace("(", "").replace(")", "").replace(" (","").replace(") ","").split(",")

                # Convert to float and scale by 1000
                # Qwen2.5vl output absolute coordinates, qwen2vl output relative coordinates
                if model_type == "qwen25vl":
                    float_numbers = []
                    for num_idx, num in enumerate(numbers):
                        num = float(num)
                        if (num_idx + 1) % 2 == 0:
                            float_numbers.append(
                                float(num / smart_resize_height))
                        else:
                            float_numbers.append(
                                float(num / smart_resize_width))
                else:
                    float_numbers = [float(num) / factor for num in numbers]

                if len(float_numbers) == 2:
                    float_numbers = [
                        float_numbers[0], float_numbers[1], float_numbers[0],
                        float_numbers[1]
                    ]
                action_inputs[param_name.strip()] = str(float_numbers)

        # import pdb; pdb.set_trace()
        actions.append({
            "reflection": reflection,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": action_inputs,
            "text": text
        })
    return actions
# def parse_action_to_structure_output(text,
#                                      factor,
#                                      origin_resized_height,
#                                      origin_resized_width,
#                                      model_type="qwen25vl",
#                                      max_pixels=16384 * 28 * 28,
#                                      min_pixels=100 * 28 * 28):
#     text = text.strip()

#     # initialize smart resize values to the original resized dimensions so they are always defined
#     smart_resize_height, smart_resize_width = origin_resized_height, origin_resized_width

#     # 解析 JSON 格式的模型输出
#     try:
#         action_data = json.loads(text)  # 直接解析模型的输出
#     except json.JSONDecodeError as e:
#         print(f"Failed to decode JSON: {e}")
#         return []

#     # 提取 reasoning 和 action 字段
#     reasoning = action_data.get("reasoning", "")
#     action_type = action_data.get("Action", "")
#     parameters = action_data.get("parameters", {})

#     # 根据 action_type 判断该执行何种操作
#     actions = []
#     if action_type == "click":
#         target_element = parameters.get("target_element", "")
#         actions.append({
#             "reasoning": reasoning,
#             "action_type": "click",
#             "action_inputs": {"target_element": target_element}
#         })
#     elif action_type == "input":
#         content = parameters.get("content", "")
#         actions.append({
#             "reasoning": reasoning,
#             "action_type": "input",
#             "action_inputs": {"content": content}
#         })
#     elif action_type == "swipe":
#         direction = parameters.get("direction", "")
#         actions.append({
#             "reasoning": reasoning,
#             "action_type": "swipe",
#             "action_inputs": {"direction": direction}
#         })
#     elif action_type == "wait":
#         actions.append({
#             "reasoning": reasoning,
#             "action_type": "wait",
#             "action_inputs": {}
#         })
#     elif action_type == "done":
#         actions.append({
#             "reasoning": reasoning,
#             "action_type": "done",
#             "action_inputs": {}
#         })
#     else:
#         print(f"Unrecognized action type: {action_type}")

#     return actions


def parsing_response_to_pyautogui_code(responses,
                                       image_height: int,
                                       image_width: int,
                                       input_swap: bool = True) -> str:
    '''
    将M模型的输出解析为OSWorld中的action，生成pyautogui代码字符串
    参数:
        response: 包含模型输出的字典，结构类似于：
        {
            "action_type": "hotkey",
            "action_inputs": {
                "hotkey": "v ctrl",
                "start_box": None,
                "end_box": None
            }
        }
    返回:
        生成的pyautogui代码字符串
    '''

    pyautogui_code = f"import pyautogui\nimport time\n"
    if isinstance(responses, dict):
        responses = [responses]
    for response_id, response in enumerate(responses):
        if "observation" in response:
            observation = response["observation"]
        else:
            observation = ""

        if "thought" in response:
            thought = response["thought"]
        else:
            thought = ""

        if response_id == 0:
            pyautogui_code += f"'''\nObservation:\n{observation}\n\nThought:\n{thought}\n'''\n"
        else:
            pyautogui_code += f"\ntime.sleep(1)\n"

        action_dict = response
        action_type = action_dict.get("action_type")
        action_inputs = action_dict.get("action_inputs", {})

        if action_type == "hotkey":
            # Parsing hotkey action
            if "key" in action_inputs:
                hotkey = action_inputs.get("key", "")
            else:
                hotkey = action_inputs.get("hotkey", "")

            if hotkey == "arrowleft":
                hotkey = "left"

            elif hotkey == "arrowright":
                hotkey = "right"

            elif hotkey == "arrowup":
                hotkey = "up"

            elif hotkey == "arrowdown":
                hotkey = "down"

            if hotkey:
                # Handle other hotkeys
                keys = hotkey.split()  # Split the keys by space
                convert_keys = []
                for key in keys:
                    if key == "space":
                        key = ' '
                    convert_keys.append(key)
                pyautogui_code += f"\npyautogui.hotkey({', '.join([repr(k) for k in convert_keys])})"

        elif action_type in ["press", "keydown"]:
            # Parsing press action
            if "key" in action_inputs:
                key_to_press = action_inputs.get("key", "")
            else:
                key_to_press = action_inputs.get("press", "")

            if key_to_press == "arrowleft":
                key_to_press = "left"

            elif key_to_press == "arrowright":
                key_to_press = "right"

            elif key_to_press == "arrowup":
                key_to_press = "up"

            elif key_to_press == "arrowdown":
                key_to_press = "down"

            elif key_to_press == "space":
                key_to_press = " "

            if key_to_press:
                # Simulate pressing a single key
                pyautogui_code += f"\npyautogui.keyDown({repr(key_to_press)})"

        elif action_type in ["release", "keyup"]:
            # Parsing press action
            if "key" in action_inputs:
                key_to_press = action_inputs.get("key", "")
            else:
                key_to_press = action_inputs.get("press", "")

            if key_to_press == "arrowleft":
                key_to_press = "left"

            elif key_to_press == "arrowright":
                key_to_press = "right"

            elif key_to_press == "arrowup":
                key_to_press = "up"

            elif key_to_press == "arrowdown":
                key_to_press = "down"

            elif key_to_press == "space":
                key_to_press = " "

            if key_to_press:
                # Simulate pressing a single key
                pyautogui_code += f"\npyautogui.keyUp({repr(key_to_press)})"

        elif action_type == "type":
            # Parsing typing action using clipboard
            content = action_inputs.get("content", "")
            content = escape_single_quotes(content)
            stripped_content = content
            if content.endswith("\n") or content.endswith("\\n"):
                stripped_content = stripped_content.rstrip("\\n").rstrip("\n")
            if content:
                if input_swap:
                    pyautogui_code += f"\nimport pyperclip"
                    pyautogui_code += f"\npyperclip.copy('{stripped_content}')"
                    pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"
                else:
                    pyautogui_code += f"\npyautogui.write('{stripped_content}', interval=0.1)"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"

        elif action_type in ["drag", "select"]:
            # Parsing drag or select action based on start and end_boxes
            start_box = action_inputs.get("start_box")
            end_box = action_inputs.get("end_box")
            if start_box and end_box:
                x1, y1, x2, y2 = eval(
                    start_box)  # Assuming box is in [x1, y1, x2, y2]
                sx = round(float((x1 + x2) / 2) * image_width, 3)
                sy = round(float((y1 + y2) / 2) * image_height, 3)
                x1, y1, x2, y2 = eval(
                    end_box)  # Assuming box is in [x1, y1, x2, y2]
                ex = round(float((x1 + x2) / 2) * image_width, 3)
                ey = round(float((y1 + y2) / 2) * image_height, 3)
                


        elif action_type == "scroll":
            # Parsing scroll action
            start_box = action_inputs.get("start_box")
            if start_box:
                x1, y1, x2, y2 = eval(
                    start_box)  # Assuming box is in [x1, y1, x2, y2]
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)

                # # 先点对应区域，再滚动
                # pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
            else:
                x = None
                y = None
            
            direction = action_inputs.get("direction", "")

            if x == None:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5)"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5)"
            else:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5, x={x}, y={y})"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5, x={x}, y={y})"

        elif action_type in [
                "click", "left_single", "left_double", "right_single", "hover"
        ]:
            # Parsing mouse click actions
            start_box = action_inputs.get("start_box")
            start_box = str(start_box)
            if start_box:
                start_box = eval(start_box)
                if len(start_box) == 4:
                    x1, y1, x2, y2 = start_box  # Assuming box is in [x1, y1, x2, y2]
                elif len(start_box) == 2:
                    x1, y1 = start_box
                    x2 = x1
                    y2 = y1
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                if action_type == "left_single" or action_type == "click":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                elif action_type == "left_double":
                    pyautogui_code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
                elif action_type == "right_single":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
                elif action_type == "hover":
                    pyautogui_code += f"\npyautogui.moveTo({x}, {y})"

        elif action_type in ["finished"]:
            pyautogui_code = f"DONE"

        else:
            pyautogui_code += f"\n# Unrecognized action type: {action_type}"

    return pyautogui_code


def add_box_token(input_string):
    # Step 1: Split the string into individual actions
    if "Action: " in input_string and "start_box=" in input_string:
        suffix = input_string.split("Action: ")[0] + "Action: "
        actions = input_string.split("Action: ")[1:]
        processed_actions = []
        for action in actions:
            action = action.strip()
            # Step 2: Extract coordinates (start_box or end_box) using regex
            coordinates = re.findall(
                r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action)

            updated_action = action  # Start with the original action
            for coord_type, x, y in coordinates:
                # Convert x and y to integers
                updated_action = updated_action.replace(
                    f"{coord_type}='({x},{y})'",
                    f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'")
            processed_actions.append(updated_action)

        # Step 5: Reconstruct the final string
        final_string = suffix + "\n\n".join(processed_actions)
    else:
        final_string = input_string
    return final_string

class ActionParser:
    """
    兼容官方 UI-TARS 的解析接口：
    输入：
      - response: 模型返回的完整文本（包含 Thought: / Action: 等）
      - image_height, image_width: 原始截图的高宽
    输出：
      - thought: str
      - raw_action: str   （形如 "click(...)" / "type(...)" 这一段）
      - parsed_action: dict，至少包含：
          - "action_type"
          - "action_params"
    """

    @staticmethod
    def parse_response(response: str, image_height: int, image_width: int):
        # 1) 先算一遍用于解析的 resize 尺寸（和训练 / 推理对齐）
        resized_h, resized_w = linear_resize(
            height=image_height,
            width=image_width,
            factor=IMAGE_FACTOR,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )

        # 2) 调用现有的解析函数，得到结构化动作列表
        actions = parse_action_to_structure_output(
            text=response,
            factor=IMAGE_FACTOR,
            origin_resized_height=resized_h,
            origin_resized_width=resized_w,
            model_type="qwen25vl",      # 你现在这版就是按 qwen2.5VL 写的
            max_pixels=MAX_PIXELS,
            min_pixels=MIN_PIXELS,
        )

        if not actions:
            # 解析失败兜底：返回 finished
            parsed_action = {
                "action_type": "finished",
                "action_params": {},
            }
            return "", "finished()", parsed_action

        a0 = actions[0]

        # 3) 取出 thought / action_type / action_inputs
        thought = a0.get("thought", "") or ""
        action_type = a0.get("action_type") or a0.get("function") or ""
        inputs = a0.get("action_inputs") or {}

        # 4) 从原始 response 中提取 'Action: ...' 这一行当 raw_action
        raw_action = ""
        for line in response.splitlines():
            if "Action:" in line:
                idx = line.find("Action:")
                raw_action = line[idx + len("Action:"):].strip()
                break
        if not raw_action:
            raw_action = response.strip()

        # 5) 把 action_inputs 映射成 framework._execute_action 需要的 action_params
        action_params: Dict[str, Any] = {}

        # 小工具：把 "[x1, y1, x2, y2]" 转成像素坐标
        def _box_to_xy(box_str: str):
            try:
                coords = ast.literal_eval(box_str)
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    xn, yn = float(coords[0]), float(coords[1])
                    # 如果看起来是 0~2 的相对坐标，就按归一化来还原
                    if 0.0 <= xn <= 2.0 and 0.0 <= yn <= 2.0:
                        x = xn * image_width
                        y = yn * image_height
                    else:
                        x, y = xn, yn
                    return float(x), float(y)
            except Exception:
                return None, None
            return None, None

        # --------- 按不同动作类型做映射 ---------
        if action_type in ("click", "left_single", "left_double", "right_single", "hover"):
            box_str = inputs.get("start_box") or inputs.get("box") or inputs.get("point")
            if box_str:
                x, y = _box_to_xy(box_str)
                if x is not None and y is not None:
                    action_params["x"] = x
                    action_params["y"] = y
            action_type_clean = "click"

        elif action_type in ("long_press",):
            box_str = inputs.get("start_box") or inputs.get("box") or inputs.get("point")
            if box_str:
                x, y = _box_to_xy(box_str)
                if x is not None and y is not None:
                    action_params["x"] = x
                    action_params["y"] = y
            action_type_clean = "long_press"

        elif action_type in ("scroll", "drag"):
            direction = (inputs.get("direction") or "down").lower()
            action_params["direction"] = direction

            box_str = inputs.get("start_box")
            if box_str:
                x, y = _box_to_xy(box_str)
                if x is not None and y is not None:
                    action_params["x"] = x
                    action_params["y"] = y
            action_type_clean = "scroll"

        elif action_type in ("type", "input"):
            content = inputs.get("content") or inputs.get("text") or ""
            action_params["text"] = content
            action_type_clean = "type"

        elif action_type in ("finished", "finish", "done"):
            action_type_clean = "finished"
            action_params = {}

        else:
            # 兜底：原样保留类型和所有参数
            action_type_clean = action_type
            for k, v in inputs.items():
                if v is not None:
                    action_params[k] = v

        parsed_action = {
            "action_type": action_type_clean,
            "action_params": action_params,
        }
        return thought, raw_action, parsed_action


def to_core_action(parsed_action: Dict[str, Any]) -> Dict[str, Any]:
    """
    把 parsed_action 变成更简洁的 core_action 格式。
    framework 里只在 on_core_action 回调里用到它，
    所以简单包一层就够了。
    """
    return {
        "type": parsed_action.get("action_type", ""),
        "params": parsed_action.get("action_params", {}),
    }