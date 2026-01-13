#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线 FSM 评估脚本：
- 不连真机，只用已有的 rawdata 轨迹建出的 AppFSM
- 每一步用 UI-TARS 的 Prompt + 模型输出的 Thought/Action
- 把模型输出解析成坐标/方向/文本动作，在 FSM 上做 state transition
"""

import os
import json
import base64
import argparse
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime  # ==== NEW: 用于生成时间戳目录 ====

from openai import OpenAI
from PIL import Image
from MobiBench.utils.task_get import get_tasks,get_tasks_1
from MobiBench.env.fsm import build_AppFSM, quick_build_AppFSM
from MobiBench.env.type_spaces import Action
from MobiBench.agents.UI_TARS.ui_tars_automation.config import MOBILE_PROMPT_TEMPLATE
import time
from MobiBench.agents.UI_TARS.ui_tars_automation.action_parser import (
    parse_action_to_structure_output,
    IMAGE_FACTOR,
    linear_resize,
)

logger = logging.getLogger(__name__)


# ---------- 一些小工具 ----------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def encode_image_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def build_messages(
    instruction: str,
    language: str,
    history: List[Dict[str, str]],
    image_data_url: str,
) -> List[Dict[str, Any]]:
    """
    用 UI-TARS 的 MOBILE_PROMPT_TEMPLATE + 历史 Thought/Action + 当前截图
    构造发给模型的 messages
    """
    base_prompt = MOBILE_PROMPT_TEMPLATE.format(
        language=language,
        instruction=instruction,
    )

    if history:
        lines = ["## Action History"]
        for i, h in enumerate(history, start=1):
            lines.append(
                f"{i}. Thought: {h['thought']}\n"
                f"   Action: {h['action']}"
            )
        history_text = "\n".join(lines)
    else:
        history_text = "## Action History\n(Empty)"

    full_text = (
        base_prompt
        + "\n\n"
        + history_text
        + "\n\nNow, based on the CURRENT screenshot below, "
          "think step by step and give the next action."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": full_text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    ]
    return messages


def call_model(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    logger.info("调用模型中…")
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content
    logger.info("模型输出（前 200 字）：%s", text.replace("\n", "\\n")[:200])
    return text


def parse_thought_action(text: str) -> Dict[str, str]:
    """
    简单从文本里提取 Thought 和 Action 字符串，交给 action_parser 再解析结构化动作。
    """
    text = text.strip()
    idx_a = text.find("Action:")
    idx_t = text.find("Thought:")

    thought = ""
    if idx_t != -1:
        if idx_a != -1 and idx_a > idx_t:
            thought = text[idx_t + len("Thought:"):idx_a].strip()
        else:
            thought = text[idx_t + len("Thought:"):].strip()

    if idx_a != -1:
        action_str = text[idx_a + len("Action:"):].strip()
        # 保留 "Thought: ...\nAction: ..." 的整体给 parse_action_to_structure_output
        full = f"Thought: {thought}\nAction: {action_str}"
    else:
        # 万一模型没写 Thought:，直接当 Action 丢进去
        full = text
        action_str = text

    return {"thought": thought, "raw_action_block": full}


def to_fsm_action(
    raw_block: str,
    img_path: str,
) -> Optional[Action]:
    """
    用 action_parser 把 "Thought:...\nAction: ..." 解析成结构化动作，
    再转成 FSM 需要的 Action(act_type, parameters)。
    """
    # 1. 读取图片尺寸，算一下 resize 后的尺寸（与训练/推理时保持一致）
    img = Image.open(img_path)
    width, height = img.size
    resized_h, resized_w = linear_resize(height, width, factor=IMAGE_FACTOR)

    # 2. 解析结构化动作
    actions = parse_action_to_structure_output(
        text=raw_block,
        factor=IMAGE_FACTOR,
        origin_resized_height=resized_h,
        origin_resized_width=resized_w,
        model_type="qwen25vl",
    )
    if not actions:
        return None

    a0 = actions[0]
    a_type = a0["action_type"]
    inputs = a0["action_inputs"] or {}

    params: Dict[str, Any] = {}

    # 3. 按 FSM 的动作空间做映射
    if a_type in ("click", "left_single", "left_double", "right_single", "hover"):
        # click 动作：inputs 里一般有 start_box，形如 "[x1_norm, y1_norm, x2_norm, y2_norm]"
        start_box_str = inputs.get("start_box")
        if start_box_str:
            try:
                box = eval(start_box_str)
                if isinstance(box, (list, tuple)) and len(box) >= 2:
                    xn, yn = float(box[0]), float(box[1])
                    # Qwen2.5VL 这里可能是归一化坐标（0~1 或 0~2），做一下判断
                    x = int(xn * width) if xn <= 2 else int(xn)
                    y = int(yn * height) if yn <= 2 else int(yn)
                    params["position_x"] = x
                    params["position_y"] = y
            except Exception:
                pass
        act_type = "click"

    elif a_type in ("scroll", "drag"):
        start_box_raw = inputs.get("start_box")
        end_box_raw = inputs.get("end_box")
        calc_direction = None

        if start_box_raw and end_box_raw:
            try:
                s_box = eval(start_box_raw) if isinstance(start_box_raw, str) else start_box_raw
                e_box = eval(end_box_raw) if isinstance(end_box_raw, str) else end_box_raw

                sx = (s_box[0] + s_box[2]) / 2
                sy = (s_box[1] + s_box[3]) / 2
                ex = (e_box[0] + e_box[2]) / 2
                ey = (e_box[1] + e_box[3]) / 2

                dx = ex - sx
                dy = ey - sy

                if abs(dx) > abs(dy):
                    calc_direction = "right" if dx > 0 else "left"
                else:
                    calc_direction = "down" if dy > 0 else "up"
                
                logger.info(f"根据坐标计算滑动方向: ({sx:.2f},{sy:.2f}) -> ({ex:.2f},{ey:.2f}) = {calc_direction}")

            except Exception as e:
                logger.warning(f"坐标计算滑动方向失败，回退到默认逻辑: {e}")

        
        final_direction = calc_direction if calc_direction else inputs.get("direction", "down")
        
        params["direction"] = final_direction.lower()
        act_type = "swipe"

    elif a_type == "type":
        text = inputs.get("content", "")
        params["text"] = text
        act_type = "input"

    elif a_type == "finished":
        # 这个就当成 done，不再在 FSM 上转移了（外层直接结束一局）
        act_type = "done"

    elif a_type in ("press_home", "press_back"):
        act_type = "back" if "back" in a_type else "home"

    else:
        # 兜底：直接把原 action_type 塞进去
        act_type = a_type

    return Action(act_type=act_type, parameters=params)


# ==== NEW: 一个简单的名字清洗函数，用于生成安全的目录名 ====
def _safe_name(text: str, max_len: int = 50) -> str:
    safe = "".join(c if c.isalnum() or c in ("_", "-", " ") else "_" for c in text)
    safe = "_".join(safe.split())  # 把空格压缩成单个下划线
    return safe[:max_len] or "task"


def run(
    fsm,
    args,
    app: str,
    task: str,
    instruction: str,
    runs_dir: str,
):

    if args.start_img_suffix:
        suffix = args.start_img_suffix.replace("\\", "/")
        found = None
        for trace in fsm.traces:
            for st in trace.states:
                if st.img_path.replace("\\", "/").endswith(suffix):
                    found = st
                    break
            if found:
                break
        if found:
            fsm.cur_state = found
            fsm.init_state = found
            fsm.history_states = [found]
            logger.info("使用指定起始状态: %s (%s)", found.img_path, found.cluster_class)
        else:
            logger.warning("没有找到匹配的起始截图 %s，改用默认 START 起点。", suffix)

    # 2. 初始化模型 client
    base_url = f"http://{args.service_ip}:{args.port}/v1"

    client = OpenAI(base_url=base_url, api_key="EMPTY")

    history_ta: List[Dict[str, str]] = []
    trace_log: List[Dict[str, Any]] = []

    done_flag = False

    # ==== NEW: 用 CLI 里的 max_steps 控制，而不是 fsm 内部默认值 ====
    max_steps = getattr(args, "max_steps", None) or getattr(fsm, "max_op_times", 20)
    fsm.max_op_times = max_steps

    for step in range(1, max_steps + 1):
        if fsm.cur_state is None:
            # 第一次调用时，fsm.action 里会自己随机选一个 START；这里先给个空动作触发
            logger.info("FSM 当前无状态，先随机初始化。")
            dummy = Action(act_type="wait", parameters={})
            fsm.action(dummy)

        cur = fsm.cur_state
        img_path = cur.img_path
        logger.info("==== Step %d | State: %s (%s) ====", step, img_path, cur.cluster_class)

        # 3. 构造 prompt + 调模型
        img_b64 = encode_image_to_data_url(img_path)
        MAX_HISTORY_STEPS = 10
        short_history = history_ta[-MAX_HISTORY_STEPS:]
        messages = build_messages(
            instruction=instruction,
            language=args.language,
            history=short_history,
            image_data_url=img_b64,
        )
        raw_output = call_model(client, args.model_name, messages)

        # 4. 提取 Thought + Action 文本
        ta = parse_thought_action(raw_output)
        thought = ta["thought"]
        raw_block = ta["raw_action_block"]

        # 5. 解析成 FSM 动作
        fsm_act = to_fsm_action(raw_block, img_path)
        if fsm_act is None:
            logger.warning("解析动作失败，结束本次评估。")
            break

        logger.info("Parsed Action: type=%s, params=%s", fsm_act.act_type, fsm_act.parameters)

        # 6. 终止判断：模型自己说 finished(...)
        if fsm_act.act_type == "done":
            logger.info("模型输出 finished(...)，结束交互。")
            # 这里不强制打断，由 FSM 状态来判断是否真正完成
            break

        # 7. 在 FSM 上执行动作
        prev_state = fsm.cur_state
        fsm.action(fsm_act)
        new_state = fsm.cur_state

        trace_log.append(
            {
                "step": step,
                "prev_img": prev_state.img_path,
                "prev_label": prev_state.cluster_class,
                "thought": thought,
                "raw_output": raw_output,
                "action_type": fsm_act.act_type,
                "action_params": fsm_act.parameters,
                "new_img": new_state.img_path,
                "new_label": new_state.cluster_class,
            }
        )
        history_ta.append({"thought": thought, "action": raw_block})

        # 8. 看看是不是到达 DONE
        if new_state.cluster_class in ("DONE", "Done", "done"):
            logger.info("到达 DONE 状态，任务完成！")
            done_flag = True
            break

        if fsm.is_failed:
            logger.info("到达 FAILED 状态，任务失败！")
            break

    result = {
        "instruction": instruction,
        "app": app,
        "task": task,
        "success": done_flag,
        "steps": trace_log,
    }

    # ========= NEW: 把结果写到 runs_dir/app/task/时间戳_指令/ 下 =========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_inst = _safe_name(instruction)
    run_dir = os.path.join(runs_dir, app, task, f"{timestamp}_{safe_inst}")
    os.makedirs(run_dir, exist_ok=True)

    # 主 JSON（全量信息）
    output_name = os.path.basename(args.output_json) if args.output_json else "fsm_eval_trace.json"
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
    print(f"评估结束，success = {done_flag}，总步数 = {len(trace_log)}")
    print(f"详细轨迹已保存到: {os.path.abspath(output_path)}")
    print(f"本次运行目录: {os.path.abspath(run_dir)}")
    if trace_log:
        print("起点截图:", trace_log[0]["prev_img"])
        print("终点截图:", trace_log[-1]["new_img"])


# ---------- 主流程 ----------

def bench():
    parser = argparse.ArgumentParser(description="Offline FSM evaluation with UI-TARS model")
    parser.add_argument("--data_root", default="/Users/fengyunfei/Desktop/mobiagent/MobiBench/data", help="MobiBench data 根目录（包含 rawdata/）")
    # parser.add_argument("--app", required=True, help="应用名称，例如 高德地图")
    # parser.add_argument("--task", required=True, help="任务名称，例如 type1")
    # parser.add_argument("--instruction", required=True, help="任务描述文本")
    parser.add_argument("--language", default="Chinese", help="Thought 使用语言")
    parser.add_argument("--service_ip",default="123.60.91.241",  help="模型服务 IP，例如 123.60.91.241")
    parser.add_argument("--port", type=int,default=9001, help="模型服务端口，例如 9001")
    parser.add_argument("--model_name", default="", help="模型名称")
    parser.add_argument("--max_steps", type=int, default=20, help="最多交互步数")
    parser.add_argument(
        "--start_img_suffix",
        default=None,
        help="指定起始状态的截图后缀，例如 '高德地图/type1/1/1.jpg' 或 '1/1.jpg'；"
             "留空则随机从 START 簇里选一个。",
    )
    parser.add_argument(
        "--output_json",
        default="fsm_eval_trace.json",
        help="单次运行的轨迹文件名（会写在每个 run 子目录中）",
    )
    parser.add_argument(
        "--runs_dir",
        default=r"/Users/fengyunfei/Desktop/mobiagent/MobiBench/agents/UI_TARS/runs",  # ==== NEW: 所有运行结果的根目录 ====
        help="所有运行结果的根目录，用于保存轨迹和坐标",
    )
    parser.add_argument("--task_json", default="/Users/fengyunfei/Desktop/mobiagent/MobiBench/data/test.json", help="task json file")
    parser.add_argument("--result_dir", default="/Users/fengyunfei/Desktop/mobiagent/MobiBench/results/dev", help="result directory")
    parser.add_argument("--log_dir", default="/Users/fengyunfei/Desktop/mobiagent/MobiBench/agents/UI_TARS/log", help="log directory")
    args = parser.parse_args()
    setup_logging()

    # 确保 runs_dir 存在
    os.makedirs(args.runs_dir, exist_ok=True)

    with open(args.task_json, 'r', encoding='utf-8') as f:
        alldata = json.load(f)

    datapath = args.data_root
    for app in alldata.keys():
        for tasktype in alldata[app]:
            tasklist = get_tasks_1(app, tasktype)
            logger.info("构建 FSM 中…")
            fsm = build_AppFSM(app=app, task=tasktype, data_path=datapath)
            # 让 FSM 内部的 max_op_times 和 CLI 一致
            fsm.max_op_times = args.max_steps

            for task in tasklist:
                print(f"任务: {task}，应用: {app}，类型: {tasktype}")
                fsm._reset()
                start = time.time()
                run(
                    fsm=fsm,
                    args=args,
                    app=app,
                    task=tasktype,
                    instruction=task,
                    runs_dir=args.runs_dir,  # ==== NEW: 传入 runs 根目录 ====
                )
                end = time.time()
                from MobiBench.utils.score_proc import save_result
                save_result(
                    md="UI-TARS",
                    app=app,
                    task=tasktype,
                    inst=task,
                    fsm=fsm,
                    time_use=end-start,
                    savepath=args.result_dir,
                )
                from MobiBench.utils.score_proc import save_visited_result
                save_visited_result(
                    md="UI-TARS",
                    app=app,
                    task=tasktype,
                    fsm=fsm,
                    savepath=args.result_dir + "/visited",
                )

if __name__ == "__main__":
    bench()
