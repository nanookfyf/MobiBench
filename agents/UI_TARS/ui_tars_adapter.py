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
import time
from openai import OpenAI
from PIL import Image
from MobiBench.utils.task_get import get_tasks
from MobiBench.env.fsm import build_AppFSM
from MobiBench.env.type_spaces import Action
from MobiBench.agents.UI_TARS.ui_tars_automation.config import MOBILE_PROMPT_TEMPLATE
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
) -> str:
    logger.info("调用模型中…")
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
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
                    x = int(xn * width) if xn <= 2 else int(xn)
                    y = int(yn * height) if yn <= 2 else int(yn)
                    params["position_x"] = x
                    params["position_y"] = y
            except Exception:
                pass
        act_type = "click"

    elif a_type in ("scroll", "drag"):
        # swipe：FSM 里叫 "swipe"，我们只需要 direction，距离目前在 _transition 里其实没用上
        direction = inputs.get("direction", "down").lower()
        params["direction"] = direction
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


def run(fsm,args,app,task,instruction):

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

    for step in range(1, fsm.max_op_times+1):
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
        messages = build_messages(
            instruction=instruction,
            language=args.language,
            history=history_ta,
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
            #done_flag = cur.cluster_class in ("DONE", "Done", "done")
            break

        if fsm.is_failed:
            logger.info("fsm env failed ，结束交互。")
            #done_flag = cur.cluster_class in ("DONE", "Done", "done")
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

    result = {
        "instruction": instruction,
        "app": app,
        "task": task,
        "success": done_flag,
        "steps": trace_log,
    }
    print("finished state",fsm.cur_state.img_path)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print()
    print(f"评估结束，success = {done_flag}，总步数 = {len(trace_log)}")
    print(f"详细轨迹已保存到: {os.path.abspath(args.output_json)}")
    if trace_log:
        print("起点截图:", trace_log[0]["prev_img"])
        print("终点截图:", trace_log[-1]["new_img"])



# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser(description="Offline FSM evaluation with UI-TARS model")
    parser.add_argument("--data_root", default="/Users/fengyunfei/Desktop/mobiagent/MobiBench/data", help="MobiBench data 根目录（包含 rawdata/）")
    parser.add_argument("--dev",default=True,help="确定是否是开发模式")
    #parser.add_argument("--app", required=True, help="应用名称，例如 高德地图")
    #parser.add_argument("--task", required=True, help="任务名称，例如 type1")
    #parser.add_argument("--instruction", required=True, help="任务描述文本")
    parser.add_argument("--language", default="Chinese", help="Thought 使用语言")
    parser.add_argument("--service_ip",default="123.60.91.241" , help="模型服务 IP，例如 123.60.91.241")
    parser.add_argument("--port", type=int, default=9001, help="模型服务端口，例如 9001")
    parser.add_argument("--model_name", default="", help="模型名称")
    parser.add_argument("--max_steps", type=int, default=20, help="最多交互步数")
    parser.add_argument("--start_img_suffix", default=None,
                        help="指定起始状态的截图后缀，例如 '高德地图/type1/1/1.jpg' 或 '1/1.jpg'；"
                             "留空则随机从 START 簇里选一个。")
    parser.add_argument("--output_json", default="fsm_eval_trace.json",
                        help="保存整条交互轨迹的 JSON 路径")

    args = parser.parse_args()
    setup_logging()

    app_list = ["小红书"]
    type_list = ["type1","type3","type5"]

    datapath = args.data_root
    for app in app_list:
        for tasktype in type_list:
            tasklist = get_tasks(app,tasktype)
            #envengine = StaticMobiAgentWorker(app,tasktype,datapath,grounder_client)
            logger.info("构建 FSM 中…")
            fsm = build_AppFSM(app=app, task=tasktype, data_path=datapath)
            for task in tasklist:
                print(f"任务: {task}，应用: {app}，类型: {tasktype}")
                fsm._reset()
                start = time.time()
                run(fsm=fsm,args=args,app=app,task=tasktype,instruction=task)
                end = time.time()
                from MobiBench.utils.score_proc import save_result
                save_result(md="UI-TARS",app=app,task=tasktype,inst=task,fsm=fsm,time_use=end-start,savepath="/Users/fengyunfei/Desktop/mobiagent/MobiBench/results/dev")
                #print(f"Bench result: steps={result['steps']}, won={result['won']}, done={result['done']}")
    
    # 1. 构建 AppFSM（会把该 app+task 的所有轨迹都吃进去）
    
    
    # 强制设置起始 state（如果指定了 start_img_suffix）
 


if __name__ == "__main__":
    main()
