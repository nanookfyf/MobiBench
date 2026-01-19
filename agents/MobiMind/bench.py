from ast import arg
from openai import OpenAI
import base64
from PIL import Image
from typing import List, Dict, Any, Optional
import json
import logging
import os
import argparse
import cv2
from MobiBench.agents.MobiMind.env_engine import StaticMobiAgentWorker
from datetime import datetime
from MobiBench.utils.task_get import get_tasks,get_tasks_1
import time 
from MobiBench.env.fsm import build_AppFSM,point_in_rectangle
from MobiBench.utils.models.text_match import semantic_similarity
from MobiBench.env.type_spaces import Action

logger = logging.getLogger(__name__)


# ---------- 一些小工具 ----------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,          # 日志级别
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # 日志格式
    )
def encode_image_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def _safe_name(text: str, max_len: int = 50) -> str:
    safe = "".join(c if c.isalnum() or c in ("_", "-", " ") else "_" for c in text)
    safe = "_".join(safe.split())  # 把空格压缩成单个下划线
    return safe[:max_len] or "task"

MAX_STEPS = 50



decider_client = None
grounder_client = None
planner_client = None

def init(service_ip, decider_port, grounder_port, planner_port):
    global decider_client, grounder_client, planner_client, general_client, general_model, apps
    decider_client = OpenAI(
        api_key = "0",
        base_url = f"http://{service_ip}:{decider_port}/v1",
    )
    grounder_client = OpenAI(
        api_key = "0",
        base_url = f"http://{service_ip}:{grounder_port}/v1",
    )
    planner_client = OpenAI(
        api_key = "sk-441155ebf2764ac78b36195e8a9978da",
        base_url = "https://api.deepseek.com",
    )

e2e_qwen3_template_v1 = """
<image>
You are a phone-use AI agent.

Please provide the next action based on the screenshot and your action history. You should do careful reasoning before providing the action.
Your action space includes:
- Name: click, Parameters: target_element (a high-level description of the UI element to click), bbox (an bounding box of the target element,[x1, y1, x2, y2]).
- Name: swipe, Parameters: direction (one of UP, DOWN, LEFT, RIGHT), start_coords (the starting coordinate [x, y]), end_coords (the ending coordinate [x, y]).
- Name: click_input, Parameters: target_element (a high-level description of the UI element to click), text (the text to input), bbox (an bounding box of the target element,[x1, y1, x2, y2]).
- Name: input, Parameters: text (the text to input).
- Name: wait, Parameters: (no parameters, will wait for 1 second).
- Name: done, Parameters: status (the completion status of the current task, one of `success`, `suspended` and `failed`).
Your output should be a JSON object with the following format:
{{"reasoning": "Your reasoning here", "action": "The next action (one of click, input, swipe, wait, done)", "parameters": {{"param1": "value1", "param2": "value2", ...}}}}

Now your task is "{task}".
Your action history is:
{history}
"""
e2e_qwen3_template_v2 = """
You are a phone-use AI agent.

Your action space includes:
- Name: click, Parameters: target_element (a high-level description of the UI element to click), bbox (an bounding box of the target element,[x1, y1, x2, y2]).
- Name: swipe, Parameters: direction (one of UP, DOWN, LEFT, RIGHT), start_coords (the starting coordinate [x, y]), end_coords (the ending coordinate [x, y]).
- Name: click_input, Parameters: target_element (a high-level description of the UI element to click), text (the text to input), bbox (an bounding box of the target element,[x1, y1, x2, y2]).
- Name: input, Parameters: text (the text to input).
- Name: wait, Parameters: (no parameters, will wait for 1 second).
- Name: done, Parameters: status (the completion status of the current task, one of `success`, `suspended` and `failed`).
Your output should be a JSON object with the following format:
{{"reasoning": "Your reasoning here", "action": "The next action (one of click, input, swipe, wait, done)", "parameters": {{"param1": "value1", "param2": "value2", ...}}}}

Now your task is "{task}".
Your action history is:
{history}
<image>
Please provide the next action based on the screenshot and your action history. You should do careful reasoning before providing the action.
"""


decider_prompt_template = """
You are a phone-use AI agent. Now your task is "{task}".
Your action history is:
{history}
Please provide the next action based on the screenshot and your action history. You should do careful reasoning before providing the action.
Your action space includes:
- Name: click, Parameters: target_element (a high-level description of the UI element to click).
- Name: swipe, Parameters: direction (one of UP, DOWN, LEFT, RIGHT).
- Name: input, Parameters: text (the text to input).
- Name: wait, Parameters: (no parameters, will wait for 1 second).
- Name: done, Parameters: (no parameters).
Your output should be a JSON object with the following format:
{{"reasoning": "Your reasoning here", "action": "The next action (one of click, input, swipe, done)", "parameters": {{"param1": "value1", ...}}}}"""

grounder_prompt_template_no_bbox = '''
Based on the screenshot, user's intent and the description of the target UI element, provide the coordinates of the element using **absolute coordinates**.
User's intent: {reasoning}
Target element's description: {description}
Your output should be a JSON object with the following format:
{{"coordinates": [x, y]}}'''

grounder_prompt_template_bbox = '''
Based on the screenshot, user's intent and the description of the target UI element, provide the bounding box of the element using **absolute coordinates**.
User's intent: {reasoning}
Target element's description: {description}
Your output should be a JSON object with the following format:
{{"bbox": [x1, y1, x2, y2]}}'''

decider_prompt_template_zh = """
你是一个手机使用AI代理。现在你的任务是“{task}”。
你的操作历史如下：
{history}
请根据截图和你的操作历史提供下一步操作。在提供操作之前，你需要进行仔细的推理。
你的操作范围包括：
- 名称：点击（click），参数：目标元素（target_element，对要点击的UI元素的高级描述）。
- 名称：滑动（swipe），参数：方向（direction，UP、DOWN、LEFT、RIGHT中的一个）。
- 名称：输入（input），参数：文本（text，要输入的文本）。
- 名称：等待（wait），参数：（无参数，将等待1秒）。
- 名称：完成（done），参数：（无参数）。
你的输出应该是一个如下格式的JSON对象：
{{"reasoning": "你的推理分析过程在此", "action": "下一步操作（click、input、swipe、done中的一个）", "parameters": {{"param1": "value1", ...}}}}"""

grounder_prompt_template_no_bbox_zh = """
根据截图、用户意图和目标UI元素的描述，使用**绝对坐标**提供该元素的坐标。
用户意图：{reasoning}
目标元素描述：{description}
你的输出应该是一个如下格式的JSON对象：
{{"coordinates": [x, y]}}"""

grounder_prompt_template_bbox_zh = """"
根据截图、用户意图和目标UI元素的描述，使用**绝对坐标**提供该元素的边界框。
用户意图：{reasoning}
目标元素描述：{description}
你的输出应该是一个如下格式的JSON对象：
{{"bbox": [x1, y1, x2, y2]}}"""

screenshot_path = "screenshot.jpg"
factor = 0.5
prices = {}

class BenchEnv:
    def __init__(self, *,app,task_type, worker, task, decider, grounder, planner,use_flag: str = "e2e_v1",
                 max_steps: int = MAX_STEPS, record_dir: str = "record",
                 run_root: str = "runs"):
        self.worker = worker
        self.task = task
        self.decider = decider
        self.grounder = grounder
        self.planner = planner
        self.max_steps = max_steps
        self.record_dir = record_dir
        self.app = app
        self.task_type = task_type
        self.use_flag = use_flag
        

        # 运行期状态
        self.history: list[str] = []
        self.actions: list[dict] = []
        self.reacts: list[dict] = []
        self.is_done = False

        # 目录与文件
        os.makedirs(self.record_dir, exist_ok=True)

        # 本次运行独立目录：runs/<YYYYmmdd-HHMMSS>-<task_slug>/
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        task_slug = "".join(ch if ch.isalnum() else "_" for ch in self.task)[:40]
        self.run_dir = os.path.join(run_root, f"{ts}-{task_slug}")
        os.makedirs(self.run_dir, exist_ok=True)

        # 常用输出文件（全部 UTF-8）
        self.file_prompts = os.path.join(self.run_dir, "prompts.txt")          # 逐步写入的 Decider prompt
        self.file_responses = os.path.join(self.run_dir, "decider_responses.jsonl")  # 原始响应（JSONL）
        self.file_trace = os.path.join(self.run_dir, "trace.jsonl")            # 规范化后的轨迹（JSONL）
        self.file_history = os.path.join(self.run_dir, "history.txt")          # 简洁历史（人读友好）
        self.file_summary = os.path.join(self.run_dir, "result.json")          # 最终汇总
        self.file_actions = os.path.join(self.run_dir, "actions.json")         # 全部规范化动作（JSON）

    # 小工具：追加文本/JSONL
    def _append_text(self, path: str, text: str):
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + ("\n" if not text.endswith("\n") else ""))

    def _append_jsonl(self, path: str, obj: dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _history_str(self) -> str:

        if not self.actions:
            return "(No history)"
        tail = self.actions[:]
        lines = []
        for i, a in enumerate(tail, 1):
            lines.append(f"{i}. " + json.dumps(a, ensure_ascii=False))
        return "\n".join(lines)

    def _save_current_img(self, step_index: int):
        """把当前状态对应截图复制到 record 里，维持原有 1.jpg, 2.jpg... 的命名"""
        img_path = self.worker.cur_state.img_path
        img = Image.open(img_path)
        save_path = os.path.join(os.getcwd(), self.record_dir, f"{step_index}.jpg")
        img.save(save_path)

    def _call_decider(self, obs_bgr_base64: str) -> dict:

        if self.use_flag == "e2e_v1":

            decider_prompt = e2e_qwen3_template_v1.format(
                task=self.task,
                history=self._history_str()
            )
            logging.info("Decider prompt:\n%s", decider_prompt)

        elif self.use_flag == "e2e_v2":
            decider_prompt = e2e_qwen3_template_v2.format(
                task=self.task,
                history=self._history_str()
            )
            logging.info("Decider prompt:\n%s", decider_prompt)
        elif self.use_flag == "decider_en":
            decider_prompt = decider_prompt_template.format(
                task=self.task,
                history=self._history_str()
            )
            logging.info("Decider prompt:\n%s", decider_prompt)
        elif self.use_flag == "decider_zh":
            decider_prompt = decider_prompt_template_zh.format(
                task=self.task,
                history=self._history_str()
            )
            logging.info("Decider prompt:\n%s", decider_prompt)
        else:
            decider_prompt = decider_prompt_template.format(
                task=self.task,
                history=self._history_str()
            )
            logging.info("Decider prompt:\n%s", decider_prompt)

        self._append_text(self.file_prompts, f"--- step {len(self.actions)+1} ---\n{decider_prompt}\n")

        resp = self.decider.chat.completions.create(
            model="",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{obs_bgr_base64}"}},
                    {"type": "text", "text": decider_prompt},
                ]
            }],
            temperature=0
        ).choices[0].message.content
        logging.info(f"Decider response:\n{resp}")
        

        try:
            self._append_jsonl(self.file_responses, json.loads(resp))
        except Exception:
            self._append_jsonl(self.file_responses, {"raw": resp})
        print(f"Decider response:\n{resp}")

        try:
            resp = json.loads(resp)
            return resp
        except:
            return False


    def _normalize_decision(self, dec: dict) -> dict:
        import re
        # 1) 规范 action：去掉括号说明 -> 小写 -> 中英映射
        a_raw = (dec.get("action") or "").strip()
        a_no_paren = re.sub(r"[（(].*?[）)]", "", a_raw)   # 删 () / （）中的内容
        a_norm = a_no_paren.strip().lower()
        action_map = {
            "点击": "click", "单击": "click", "click": "click",
            "输入": "input", "input": "input",
            "滑动": "swipe", "上滑": "swipe", "下滑": "swipe", "左滑": "swipe", "右滑": "swipe", "swipe": "swipe",
            "等待": "wait", "wait": "wait",
            "完成": "done", "结束": "done", "done": "done", "停止": "done"
        }
        action = action_map.get(a_norm, a_norm)

        # 2) 规范参数键名
        p = dec.get("parameters", {}) or {}
        key_map = {
            "目标元素": "target_element", "元素": "target_element",
            "文本": "text", "内容": "text",
            "方向": "direction",
            "起点": "start", "终点": "end",
            "坐标": "coords",
            "横坐标": "x", "纵坐标": "y",
            "X": "x", "Y": "y"
        }
        norm = {key_map.get(k, k): v for k, v in p.items()}

        # 3) 方向值统一
        if isinstance(norm.get("direction"), str):
            d = norm["direction"].strip()
            dir_map = {"上": "UP", "下": "DOWN", "左": "LEFT", "右": "RIGHT",
                    "up": "UP", "down": "DOWN", "left": "LEFT", "right": "RIGHT"}
            norm["direction"] = dir_map.get(d, d)

        return {"reasoning": dec.get("reasoning", ""), "action": action, "parameters": norm}

    def _valid_decision(self, dec: dict) -> bool:
        a = dec.get("action")
        p = dec.get("parameters", {}) or {}
        need = {
            "click": ["target_element"],
            "input": ["text"],       
            "swipe": ["direction"],
            "wait": [],
            "done": [],
        }
        must = need.get(a, [])
        for k in must:
            if k not in p or p[k] in (None, ""):
                logging.warning("[decider] missing key for action=%s: %s in %s", a, k, p)
                return False
        return True
    
    def validate_action_parameters(decider_response):
        """
        校验不同动作的字段完整性
        
        Args:
            decider_response: 解析后的 JSON 响应字典
        
        Raises:
            ValueError: 当必需字段缺失时
        """

        action = decider_response.get("action")
        parameters = decider_response.get("parameters", {})
        
        if not action:
            raise ValueError("Missing required field: 'action'")
        
        if not decider_response.get("reasoning"):
            raise ValueError("Missing required field: 'reasoning'")
        
        # 根据不同动作类型校验必需参数
        if action == "click":
            if not parameters.get("target_element"):
                raise ValueError("Click action missing required parameter: 'target_element'")
            # e2e模式下需要校验bbox
            # 注意：这里不直接检查bbox，因为可能在非e2e模式下不需要
        elif action == "click_input":
            if not parameters.get("target_element"):
                raise ValueError("Click_input action missing required parameter: 'target_element'")
            if not parameters.get("bbox"):
                raise ValueError("Click_input action missing required parameter: 'bbox'")
            if not parameters.get("text"):
                raise ValueError("Click_input action missing required parameter: 'text'")
            
        elif action == "input":
            if "text" not in parameters:
                raise ValueError("Input action missing required parameter: 'text'")
            # text可以为空字符串，所以只检查是否存在该字段
        
        elif action == "swipe":
            direction = parameters.get("direction")
            if not direction:
                raise ValueError("Swipe action missing required parameter: 'direction'")
            if direction.upper() not in ["UP", "DOWN", "LEFT", "RIGHT"]:
                raise ValueError(f"Invalid swipe direction: '{direction}'. Must be one of: UP, DOWN, LEFT, RIGHT")
        
        elif action == "done":
            status = parameters.get("status")
            if not status:
                raise ValueError("Done action missing required parameter: 'status'")
        
        elif action == "long_press":
            if not parameters.get("target_element"):
                raise ValueError("Long_press action missing required parameter: 'target_element'")
        
        elif action == "open_app":
            if not parameters.get("app_name"):
                raise ValueError("Open_app action missing required parameter: 'app_name'")
        
        elif action == "wait":
            # wait动作通常不需要额外参数
            pass
        
        else:
            raise ValueError(f"Unknown action: '{action}'")
        
        return True

    def bench(self) -> dict:
        """
        执行完整基准流程：
        - 每步：取观测、存截图、问 decider、worker.step、更新历史
        - 终止：done==True 或达到 max_steps
        返回：结果字典（actions/reacts/history/胜负标志等）
        """
        step = 1
        won = 0
        is_filed = False
        trace_log = []

        while step <= self.max_steps and not self.is_done and not is_filed:
            # 1) 取观测
            obs_rgb = self.worker._get_obs()        
            # 转成 base64 给 decider
            _, buf = cv2.imencode(".jpg", obs_rgb[:, :, ::-1])  
            obs_b64 = base64.b64encode(buf).decode("utf-8")

            # 2) 保存当前环境截图（和旧脚本一致：1.jpg, 2.jpg,...）
            #self._save_current_img(step + 1)

           
            # 3) 调 decider + 规范化
            dec_raw = self._call_decider(obs_b64)
            if dec_raw is False:
                print("!![decoder err break]!!")
                break
            #print(dec_raw)
            dec = self._normalize_decision(dec_raw)
            #dec = dec_raw
            #self.validate_action_parameters(dec)

            self.actions.append(dec)
            
            self._append_jsonl(self.file_trace, {
            "step": step + 1,
            "dec": dec,
            "phase": "pre_exec"
            })
            # 新增：简单 schema 校验，缺关键参数就跳过该步
            if not self._valid_decision(dec):
                # 记录一条“跳过”的历史，避免下一轮模型完全不知道刚才发生了啥
                self.history.append(f"skip -> invalid decision for action={dec.get('action')}")
                step += 1
                continue


            # 若是 done，直接结束（不要再丢给 worker）
            
                #self.is_done = True
                #won = 1  # 如果你的定义里 done=成功；不需要就改为 0
                #break

            # 4) 兼容 reacts 结构（保留原有格式）
            converted = {
                "reasoning": dec["reasoning"],
                "function": {"name": dec["action"], "parameters": dec["parameters"]}
            }
            self.reacts.append(converted)

            # 5) 真正执行一步
            prev_state = self.worker.fsm.cur_state
            obs_next, reward, done, info,is_filed,stdact = self.worker.step(dec)
            new_state = self.worker.fsm.cur_state

            trace_log.append(
            {
                "step": step,
                "prev_img": prev_state.img_path,
                "prev_label": prev_state.cluster_class,
                "thought": dec["reasoning"],
                "raw_output": dec_raw,
                "action_type": stdact.act_type,
                "action_params": stdact.parameters,
                "new_img": new_state.img_path,
                "new_label": new_state.cluster_class,
            }
            )

            self.is_done = bool(done)
            won = info.get("won", 0)
            if dec["action"] == "done":
                self.history.append("done")
                is_filed = True

            # 6) 更新文字历史（给下一轮 prompt 用）
            a = dec["action"]
            p = dec.get("parameters", {})
            if a == "click":
                self.history.append(f'click -> {p.get("target_element","")}')
            elif a == "input":
                self.history.append(f'input -> "{p.get("text","")}"')
            elif a == "swipe":
                self.history.append(f'swipe -> {p.get("direction","")}')
            elif a == "wait":
                self.history.append("wait")
            elif a== "click_input":
                self.history.append(f'click_input -> {p.get("bbox")} {p.get("text")}')
            else:
                self.history.append("None")

            self._append_text(self.file_history, self.history[-1])
            step += 1

        
        print("finished state",self.worker.fsm.cur_state.img_path)
        result = {
            "instruction": self.task,
            "app": self.app,
            "task": self.task_type,
            "success": self.is_done,
            "steps": trace_log,
        }

        # ========= NEW: 把结果写到 runs_dir/app/task/时间戳_指令/ 下 =========
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_inst = _safe_name(self.task)
        run_dir = os.path.join(self.record_dir, self.app, self.task_type, f"{timestamp}_{safe_inst}")
        os.makedirs(run_dir, exist_ok=True)

        # 主 JSON（全量信息）
        output_name = os.path.basename("fsm_eval_trace.json") 
        output_path = os.path.join(run_dir, output_name)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 精简版 actions.json：只保留一步一步的动作要素 + 图像路径
        simple_actions = []
        for s in trace_log:
            simple_actions.append(
                {
                    "step": s["step"],
                    "action_type": s["action_type"],
                    "action_params": s["action_params"],
                    "prev_img": s["prev_img"],
                    "new_img": s["new_img"],
                }
            )
        with open(os.path.join(run_dir, "actions.json"), "w", encoding="utf-8") as f:
            json.dump(simple_actions, f, ensure_ascii=False, indent=2)

        # clicks.json：专门导出点击坐标，便于分析
        click_actions = [
            {
                "step": s["step"],
                "position_x": s["action_params"].get("position_x"),
                "position_y": s["action_params"].get("position_y"),
                "prev_img": s["prev_img"],
                "new_img": s["new_img"],
            }
            for s in trace_log
            if s["action_type"] == "click"
        ]
        if click_actions:
            with open(os.path.join(run_dir, "clicks.json"), "w", encoding="utf-8") as f:
                json.dump(click_actions, f, ensure_ascii=False, indent=2)

        # 控制台打印总结
        print()
        print(f"评估结束，success = {self.is_done}，总步数 = {len(trace_log)}")
        print(f"详细轨迹已保存到: {os.path.abspath(output_path)}")
        print(f"本次运行目录: {os.path.abspath(run_dir)}")
        if trace_log:
            print("起点截图:", trace_log[0]["prev_img"])
            print("终点截图:", trace_log[-1]["new_img"])
            
    
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="MobiMind Agent")
    parser.add_argument("--service_ip", type=str, default="123.60.91.241", help="Ip for the services (default: localhost)")
    parser.add_argument("--decider_port", type=int, default=9003, help="Port for decider service (default: 8000)")
    parser.add_argument("--grounder_port", type=int, default=9003, help="Port for grounder service (default: 8001)")
    parser.add_argument("--planner_port", type=int, default=8000, help="Port for planner service (default: 8002)")
    parser.add_argument("--datapath", type=str, default="/Users/fengyunfei/Desktop/mobiagent/MobiBench/data", help="path to data")
    parser.add_argument("--task_json", default="/Users/fengyunfei/Desktop/mobiagent/MobiBench/data/test.json", help="task json file")
    parser.add_argument("--result_dir", default="/Users/fengyunfei/Desktop/mobiagent/MobiBench/results/dev", help="result directory")
    parser.add_argument("--log_dir", default="/Users/fengyunfei/Desktop/mobiagent/MobiBench/agents/MobiMind/log", help="log directory")
    #parser.add_argument("--prompt_file", default="e2e_v1", help="chose prompt file")
    parser.add_argument("--e2e", choices=["on", "off"], default="on", help="whether use e2e mode")
    #parser.add_argument("--use_qwen3", choices=["on", "off"], default="on", help="Whether to use Qwen3VL-based model (default: on)")
    args = parser.parse_args()

    # 使用命令行参数初始化
    init(args.service_ip, args.decider_port, args.grounder_port, args.planner_port)

    #device = AndroidDevice()
    #print(f"connect to device")

    # data_base_dir = os.path.join(os.path.dirname(__file__), 'data')
    # if not os.path.exists(data_base_dir):
    #     os.makedirs(data_base_dir)
    # task_json_path = os.path.join(os.path.dirname(__file__), "task.json")
    with open(args.task_json, 'r', encoding='utf-8') as f:
        alldata = json.load(f)

    use_flag = "decider_en"
    if args.e2e == "on":
        use_flag = "e2e_v1"
    elif args.e2e == "off":
        use_flag = "decider_en"

    datapath = args.datapath
    for app in alldata.keys():
        for tasktype in alldata[app]:
            tasklist = get_tasks(app,tasktype)
            envengine = StaticMobiAgentWorker(app,tasktype,datapath,grounder_client,use_flag = args.agent_mode)
            for task in tasklist:
                print(f"任务: {task}，应用: {app}，类型: {tasktype}")
                envengine.reset()
                runner = BenchEnv(
                        app=app,
                        task_type=tasktype,
                        worker=envengine,
                        task=task,
                        decider=decider_client,
                        grounder=grounder_client,
                        planner=planner_client,
                        max_steps=MAX_STEPS,
                        record_dir=args.log_dir,  
                        use_flag=use_flag
                    )
                start = time.time()
                result = runner.bench()
                end = time.time()
                from MobiBench.utils.score_proc import save_result
                save_result(md="MobiMind",app=app,task=tasktype,inst=task,fsm=envengine.fsm,time_use=end-start,savepath=args.log_dir)
                from MobiBench.utils.score_proc import save_env_result
                save_env_result(
                    app=app,
                    task=tasktype,
                    fsm=envengine.fsm,
                    savepath=args.result_dir + "/env",
                )
                from MobiBench.utils.score_proc import save_visited_result
                save_visited_result(
                    md="MobiMind",
                    app=app,
                    task=tasktype,
                    fsm=envengine.fsm,
                    savepath=args.result_dir + "/visited",
                )


   