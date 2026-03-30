# -*- coding: utf-8 -*-
import os
import re
import json
import math
import random
from typing import List, Dict, Optional, Any, Union

# 以包方式运行：python -m MobiFlow.static_bench.multi_trace_to_one
from .type_spaces import *          # 依赖: State, Action 等
from .parsedata2link import *       # 依赖: TraceParser, TraceLink, merge_info


# ----------------------------
# 工具函数
# ----------------------------

def union_maps(map1, map2):
    """
    合并两个状态的 map_info 字典: map1 -合并给> map2 【不覆盖】
    """
    for act_type in map1.keys():
        if act_type in map2.keys():
            for k, v in map1[act_type].items():
                if k not in map2[act_type].keys():
                    map2[act_type][k] = v
    return map2


def random_choice_from_list(lst):
    """
    从列表中随机选择一个元素
    """
    if not lst:
        raise ValueError("列表不能为空")
    return random.choice(lst)


def loadactions(file_path: str) -> List[Action]:
    """
    从 actions.json 中加载动作序列（包装为 Action）
    """
    Actions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        jsonf = json.load(f)
        actions = jsonf.get("actions", [])
        for action in actions:
            act_type = action.get("type", "")
            act_param = {}
            for k, v in action.items():
                if k != "type" and v is not None:
                    act_param[k] = v
            action_obj = Action(act_type=act_type, parameters=act_param)
            Actions.append(action_obj)
    return Actions


def load_actions_raw(file_path: str) -> list:
    """
    返回 actions.json 中的原始 actions 列表（不包装为 Action）
    """
    with open(file_path, "r", encoding="utf-8") as f:
        jf = json.load(f)
    return jf.get("actions", []) or []


def assign_labels_to_state_indices(actions_raw: list, num_states: int) -> Dict[int, str]:

    labels_by_state: Dict[int, str] = {}
    curr_idx = 0
    FRAME_ADVANCE_TYPES = {"click", "swipe", "input", "wait"}
    for act in actions_raw:
        t = (act or {}).get("type", "")
        if t == "tag":
            lbl = act.get("label")
            if lbl is not None:
                labels_by_state[curr_idx] = str(lbl)
            continue
        if t in FRAME_ADVANCE_TYPES:
            curr_idx = min(curr_idx + 1, num_states - 1)
    return labels_by_state


def point_in_rectangle(x: float, y: float, x1, y1, x2, y2) -> bool:
    return x1 <= x <= x2 and y1 <= y <= y2


def _num_from_path(p: str) -> int:
    """
    提取路径里最后出现的数字（用于对帧按数字顺序排序）
    """
    m = re.findall(r'(\d+)', os.path.basename(p))
    return int(m[-1]) if m else 10**9


# ----------------------------
# AppFSM
# ----------------------------

class AppFSM:

    def __init__(self, app: str, task: str, data_path: str) -> None:
        self.app = app                   # app name
        self.task = task                 # task name
        self.data_path = data_path       # 数据根路径（到 app 上一级目录）
        self.filename = os.path.join(data_path, "fsm", app, task, "fsm_traces.json")

        self.traces: List[TraceLink] = []      # 多条 TraceLink
        self.hash_map: Dict[str, State] = {}   # img_path -> State
        self.app_states: Dict[str, List[State]] = {}  # cluster_key -> [State,...]

        self.cur_state: Optional[State] = None
        self.parser = TraceParser()

        # 参数与计数
        self.max_op_times = 100
        self.max_undefine_op_times = 5
        self.undefine_op_times = 0
        self.op_times = 0
        
        # 入口：解析 -> 聚簇 -> 归约
        self._init_states()
        self._cluster()
        self._reduce_transitions()

        self.max_trace_step = max( len(t) for t in self.traces)
        self.min_trace_step = min( len(t) for t in self.traces)

    # ---------- I/O ----------

    def save_traces(self):
        """
        保存多条轨迹链到文件（与 parsedata.py 输出格式对齐）
        """
        directory = os.path.dirname(self.filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        data = {
            "version": "1.0",
            "total_traces": len(self.traces),
            "traces": [trace.to_dict() for trace in self.traces]
        }
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def load_traces(self):
        """
        从文件加载多条轨迹链
        """
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                traces = [TraceLink.from_dict(trace) for trace in data["traces"]]
                self.traces = traces
            return
        except FileNotFoundError:
            return

    # ---------- 构建 ----------

    def _init_states(self):
        """
        解析 data_path/app/task 下的所有轨迹目录：
         - parse_trace_directory(root)
         - 对帧按“数字”排序（避免 10.jpg < 2.jpg 的问题）
         - 读取 root/actions.json，挂到 trace_link.actions
         - 按“会截屏动作推进、tag 不推进”的规则将 tag 映射到帧索引
         - merge_info(trace_link) 将帧级转移写入各状态的 map_info
        """
        task_path = os.path.join(self.data_path, self.app, self.task)
        trace_links: List[TraceLink] = []

        walker = os.walk(task_path)
        next(walker, None)  # 跳过根目录

        for root, _, _ in walker:
            try:
                # 1) 解析单条轨迹
                trace_link = self.parser.parse_trace_directory(root)

                # 1.1) 对帧按数字顺序排序（就地）
                trace_link.states.sort(key=lambda s: _num_from_path(getattr(s, "img_path", "")))

                # 2) 读取该轨迹的 actions.json 原始动作 & 包装后的动作
                actions_json_path = os.path.join(root, "actions.json")
                if os.path.isfile(actions_json_path):
                    actions_raw = load_actions_raw(actions_json_path)
                    trace_link.actions = loadactions(actions_json_path)
                else:
                    actions_raw = []
                    trace_link.actions = []

                # 3) 把 tag 映射到帧序号，写入到对应 State 的 cluster_class（仅对有标签的帧）
                labels_by_state = assign_labels_to_state_indices(actions_raw, len(trace_link.states))
                for idx, state in enumerate(trace_link.states):
                    if idx in labels_by_state:
                        state.cluster_class = labels_by_state[idx]  # 有 label
                    elif idx == 0 and not getattr(state, "cluster_class", None):
                        state.cluster_class = "START"               # 起点默认 START
                    elif idx == len(trace_link.states) - 1 and not getattr(state, "cluster_class", None):
                        state.cluster_class = "DONE"                # 终点默认 Done
                    else:
                        # 未标注：保持为 None（不要写成 "Unlabeled"）
                        if getattr(state, "cluster_class", None) == "Unlabeled":
                            state.cluster_class = None

                # 4) 将帧级转移写入各状态 map_info（使用我们刚挂上的 trace_link.actions）
                merge_info(trace_link)

                # 5) 纳入全局
                trace_links.append(trace_link)

                # 解析器应当维护 states_map: img_path -> State
                if hasattr(self.parser, "states_map"):
                    self.hash_map.update(self.parser.states_map)

            except Exception as e:
                print(f"Error parsing trace in {root}: {e}")
                continue

        self.traces = trace_links

    def _cluster(self):
        """
        仅对“有标签”的帧进行分簇；未标注帧（cluster_class is None）→ 每帧自成一簇。
        这个策略能保证：只有你打过 tag 的状态才会合并；其余中间帧都会保留为独立节点。
        """
        print("all cluster ...")
        self.app_states.clear()

        from pathlib import Path

        for t in self.traces:
            for s in t.states:
                raw = getattr(s, "cluster_class", None)
                if raw and raw != "Unlabeled":
                    # 有显式标签（包含 START）→ 用标签名做簇键
                    key = raw
                else:
                    # 未标注：一个状态一个簇，用“目录名/文件名”做唯一 key（短且唯一）
                    p = Path(s.img_path)
                    key = f"__UNLABELED__::{p.parent.name}/{p.stem}"
                    # 显式记为未标注
                    s.cluster_class = "unlabeled"
                self.app_states.setdefault(key, []).append(s)

    def _reduce_transitions(self):
        """
        对每个簇归约转移：
          - 有标签的簇：做并集，得到“标签级”的统一 map_info（不覆盖已有键）
          - 未标注簇(__UNLABELED__::开头)：不做并集，保留各自原始 map_info
        """
        print("reduce transitions ...")
        for key, states in self.app_states.items():
            # map_info 键兜底
            for s in states:
                for k in ("click", "swipe", "input", "wait"):
                    s.map_info.setdefault(k, {})

            # 未标注：一个状态一个簇，跳过并集
            if isinstance(key, str) and key.startswith("__UNLABELED__::"):
                continue

            # 有标签：做并集
            all_map = {"click": {}, "swipe": {}, "input": {}, "wait": {}}
            for s in states:
                for act_type, map_ in s.map_info.items():
                    if act_type in all_map:
                        all_map[act_type].update(map_)
            for s in states:
                union_maps(all_map, s.map_info)

    # ---------- 运行/模拟 ----------

    def _transition(self, act: Action) -> State:
        """
        给定一个动作，从当前状态根据 map_info 转移；若未匹配则留在原状态。
        """
        self.op_times += 1

        if act.act_type == "click":
            for k, v in self.cur_state.map_info.get("click", {}).items():
                print("check click block:",k)
                if k == "unknown":
                    continue
                # k 可能是 tuple(x1,y1,x2,y2)；若不是，跳过
                if isinstance(k, (list, tuple)) and len(k) == 4:
                    if point_in_rectangle(
                        act.parameters.get('position_x', -1), act.parameters.get('position_y', -1),
                        k[0], k[1], k[2], k[3]
                    ):
                        return self.hash_map.get(v, self.cur_state)
            self.undefine_op_times += 1
            return self.cur_state

        elif act.act_type == "swipe":
            for k, v in self.cur_state.map_info.get("swipe", {}).items():
                if not (isinstance(k, (list, tuple)) and len(k) == 2):
                    continue
                dir_ = act.parameters.get("direction")
                dis = 0
                if dir_ in ("up", "down"):
                    dis = abs(act.parameters.get("press_position_y", 0) - act.parameters.get("release_position_y", 0))
                elif dir_ in ("left", "right"):
                    dis = abs(act.parameters.get("press_position_x", 0) - act.parameters.get("release_position_x", 0))
                # k: (direction, distance_value)
                if k[0] == dir_ and (k[1] - 2) <= dis <= (k[1] + 2):
                    return self.hash_map.get(v, self.cur_state)
            self.undefine_op_times += 1
            return self.cur_state

        elif act.act_type == "input":
            for k, v in self.cur_state.map_info.get("input", {}).items():
                if k == act.parameters.get("text"):
                    return self.hash_map.get(v, self.cur_state)
            self.undefine_op_times += 1
            return self.cur_state

        else:
            # 其它动作（如 wait、unknown 等），当前未定义具体转移：保持不变
            self.undefine_op_times += 1
            return self.cur_state

    def _reset(self):
        if "START" in self.app_states and self.app_states["START"]:
            self.cur_state = random_choice_from_list(self.app_states["START"])
        else:
            self.cur_state = random_choice_from_list(self.app_states[1])
        self.undefine_op_times = 0
        self.op_times = 0

    def action(self, act: Action) -> State:
        """
        在 FSM 上执行一个动作，返回新状态。
        """
        if self.cur_state is None:
            # 优先用 START 簇作为起点；否则任选一个簇
            if "START" in self.app_states and self.app_states["START"]:
                self.cur_state = random_choice_from_list(self.app_states["START"])
            else:
                any_list = next(iter(self.app_states.values()))
                self.cur_state = random_choice_from_list(any_list)
            print(f"random init state {self.cur_state.img_path} {self.cur_state.cluster_class}...")

        self.cur_state = self._transition(act)
        return self.cur_state


def build_AppFSM(app: str, task: str, data_path: str) -> AppFSM:
    """
    构建 AppFSM 的便捷函数
    """
    fsm = AppFSM(app, task, data_path)
    return fsm  



# ----------------------------
# main
# ----------------------------

if __name__ == "__main__":
    # 示例：改成你的实际路径
    # data_path 指向“应用父目录”，不要把 app 拼进去
    data_path = r"/Users/fff/Desktop/mobiagent/Mobibench/data"
    app = "美团"
    task = "type1"

    fsm = AppFSM(app, task, data_path)
    fsm.save_traces()
    # 自动挑一条轨迹进行“回放验证”（可选）
    import glob
    cands = glob.glob(os.path.join(data_path, app, task, "*", "actions.json"))
    actions_path = sorted(cands)[0] if cands else None

    if actions_path and os.path.isfile(actions_path):
        actions = loadactions(actions_path)
        fsm.save_traces()  # 输出 fsm_traces.json（与 parsedata.py 输出结构一致）

        for a in actions:
            st = fsm.action(a)
            print(f"action: {a}, state: {st.img_path} {st.cluster_class}")
            if getattr(st, "cluster_class", None) == "Done":
                print("task done!")
                break
    else:
        # 若未指定某条轨迹，至少保存一次解析到的多条轨迹
        fsm.save_traces()
        print("No specific actions.json for replay; saved aggregated traces instead.")
