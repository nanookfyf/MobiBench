# -*- coding: utf-8 -*-
import os
import re
import json
import random
import hashlib
from datetime import datetime
import pickle
from typing import List, Dict, Optional
from MobiBench.env.type_spaces import *          # 依赖: State, Action 等
from MobiBench.env.parsedata2link import *       # 依赖: TraceParser, TraceLink, merge_info ,set_scores
from MobiBench.utils.models.text_match import semantic_similarity


class ScoreDeclare:
    level_score = 90
    done_score =  10
    op_times_penalty = 5
ScoreDeclare = ScoreDeclare()
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

def reassign_values(data):
    # 按值排序，获取排序后的键
    sorted_keys = sorted(data.keys(), key=lambda k: data[k])
    
    # 创建新的字典，分配从0开始的连续值
    new_data = {}
    for index, key in enumerate(sorted_keys):
        new_data[key] = index
    
    return new_data

def _num_from_path(p: str) -> int:
    """
    提取路径里最后出现的数字（用于对帧按数字顺序排序）
    """
    m = re.findall(r'(\d+)', os.path.basename(p))
    return int(m[-1]) if m else 10**9

def _cal_level_score(cluster_level,cluster):

    score = ScoreDeclare.level_score
    max_k = None
    max_v = 0
    for k,v in cluster_level.items():
        if v> max_v:
            max_v = v 
            max_k = k 
    
    cluster_v = cluster_level[cluster]
    score = (max_v - cluster_v)* score / max_v
    return score



# ----------------------------
# AppFSM
# ----------------------------

class AppFSM:

    def __init__(self, app: str, task: str, data_path: str,is_init = True,use_cache: bool = True, cache_dir: str = None) -> None:
        self.app = app                   # app name
        self.task = task                 # task name
        self.data_path = data_path       # 数据根路径（到 app 上一级目录）
        self.filename = os.path.join(data_path, "fsm", app, task, "fsm_traces.json")
        self.use_cache = use_cache       # 是否使用缓存
        self.cache_dir = cache_dir       # 缓存目录

        self.traces: List[TraceLink] = []      # 多条 TraceLink
        self.hash_map: Dict[str, State] = {}   # img_path -> State
        self.app_states: Dict[str, List[State]] = {}  # cluster_key -> [State,...]
        self.cluster_level = {} # cluster_key -> int (当前状态类 距离DONE的 最短距离)
        

        self.cur_state: Optional[State] = None
        self.init_state: Optional[State] = None
        self.history_states: List[State] = []

        self.parser = TraceParser()


        # 是否强制退出
        self.is_failed = False

        # 参数与计数
        self.max_op_times = 30
        self.max_undefine_op_times = 5
        self.undefine_op_times = 0
        self.op_times = 0

        #scores
        self.score = 0.0
        self._lock = False
        self.cluster_level["DONE" ] = 0 
        # 计算数据哈希
        #self.data_hash = get_data_hash(data_path, app, task)
        
        
        print(f"从原始数据构建 {app}/{task}")
        self._init_states()
        self._cluster()
        self._reduce_transitions()
        self.max_trace_step = max(len(t) for t in self.traces) if self.traces else 0
        self.min_trace_step = min(len(t) for t in self.traces) if self.traces else 0

        self.visited_trace = []
        self.visited = {k: False for k in self.hash_map.keys()}
        
        
        print("cluster class's level to Done\n",self.cluster_level)

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
                        if state.cluster_class not in self.cluster_level.keys():
                            self.cluster_level[state.cluster_class] = len(trace_link.states) - idx 
                        else:
                            self.cluster_level[state.cluster_class] = max(len(trace_link.states) - idx ,self.cluster_level[state.cluster_class])

                    elif idx == 0 and not getattr(state, "cluster_class", None):
                        state.cluster_class = "START"               # 起点默认 START
                        if state.cluster_class not in self.cluster_level.keys():
                            self.cluster_level[state.cluster_class] = len(trace_link.states) - idx 
                        else:
                            self.cluster_level[state.cluster_class] = max(len(trace_link.states) - idx ,self.cluster_level[state.cluster_class])


                    elif idx == len(trace_link.states) - 1 and not getattr(state, "cluster_class", None):
                        state.cluster_class = "DONE"                # 终点默认 Done
                    else:
                        # 未标注：保持为 None（不要写成 "Unlabeled"）
                        if getattr(state, "cluster_class", None) == "Unlabeled":
                            state.cluster_class = None
                self.cluster_level = reassign_values(self.cluster_level)
                # 4) 将帧级转移写入各状态 map_info（使用我们刚挂上的 trace_link.actions）
                merge_info(trace_link)
                # 5) 根据link为每一个state的score属性给值
                set_scores(trace_link)

                # 6) 纳入全局
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
                print("Checking click area:",k,"-->",v)

                if k == "unknown":
                    continue
                # k 可能是 tuple(x1,y1,x2,y2)；若不是，跳过
                if isinstance(k, (list, tuple)) and len(k) == 4:
                    if point_in_rectangle(
                        act.parameters.get('position_x', -1), act.parameters.get('position_y', -1),
                        k[0], k[1], k[2], k[3]
                    ):
                        self.undefine_op_times = 0
                        return self.hash_map.get(v, self.cur_state)
            self.undefine_op_times += 1
            return self.cur_state

        elif act.act_type == "swipe":

            for k, v in self.cur_state.map_info.get("swipe", {}).items():
                print("Checking swipe ",k)
                dir_ = act.parameters.get("direction")
                if k[0] == dir_ :
                    self.undefine_op_times = 0
                    return self.hash_map.get(v, self.cur_state)
                
            self.undefine_op_times += 1
            return self.cur_state


        elif act.act_type == "input":

            for k, v in self.cur_state.map_info.get("input", {}).items():
                print("Checking input text:",k)
                if k == act.parameters.get("text"):
                    self.undefine_op_times = 0
                    return self.hash_map.get(v, self.cur_state)
                elif  semantic_similarity(k,act.parameters.get("text"))['cosine_similarity']>0.7:
                    self.undefine_op_times = 0
                    return self.hash_map.get(v, self.cur_state)
            self.undefine_op_times += 1
            return self.cur_state

        elif act.act_type == "click_input":
            # stage-1 click
            tmp_state = None 
            for k, v in self.cur_state.map_info.get("click", {}).items():
                print("[click-input] Checking click area:",k,"-->",v)
                if k == "unknown":
                    continue
                # k 可能是 tuple(x1,y1,x2,y2)；若不是，跳过
                if isinstance(k, (list, tuple)) and len(k) == 4:
                    if point_in_rectangle(
                        act.parameters.get('position_x', -1), act.parameters.get('position_y', -1),
                        k[0], k[1], k[2], k[3]
                    ):
                        tmp_state =  self.hash_map.get(v, self.cur_state)
            
            if tmp_state == None :
                self.undefine_op_times += 1

                return self.cur_state
            else:
                # stage2 
                # click_input的click能跳转 也跳转一半
                self.cur_state = tmp_state # 【支持一半的转移】
                for k, v in tmp_state.map_info.get("input", {}).items():
                    print("[click-input] Checking input text:",k)
                    if k == act.parameters.get("text"):
                        self.undefine_op_times = 0
                        return self.hash_map.get(v, self.cur_state)
                    elif  semantic_similarity(k,act.parameters.get("text"))['cosine_similarity']>0.7:
                        self.undefine_op_times = 0
                        return self.hash_map.get(v, self.cur_state)
                self.undefine_op_times += 1
            return self.cur_state
        
        elif act.act_type == "home":
            print("Going home...")
            self.undefine_op_times = 0
            self.cur_state = random_choice_from_list(self.app_states["START"])  # home 回退到初始状态
            self.history_states = [self.cur_state]
            
            return self.cur_state
        
        elif act.act_type == "back":
            print("Going back...")
            if len(self.history_states) >= 2:
                self.history_states.pop()  # 弹出当前状态
                self.cur_state = self.history_states[-1]  # 回到上一个状态
                self.undefine_op_times = 0
            if len(self.history_states)<=1:
                self.cur_state = random_choice_from_list(self.app_states["START"])
                self.history_states = [self.cur_state]
            
            return self.cur_state
            
        elif act.act_type == "wait":
            print("Waiting...")
            for k, v in self.cur_state.map_info.get("wait", {}).items():
                print ("Checking wait ",k)  
                self.undefine_op_times += 1
                return self.hash_map.get(v, self.cur_state)
            self.undefine_op_times += 1
            return self.cur_state
        else:
            # 其它动作（如 wait、unknown 等），当前未定义具体转移：保持不变
            self.undefine_op_times += 1
            return self.cur_state

    def _reset(self):
        """
            重置该app fsm
        """

        if "START" in self.app_states and self.app_states["START"]:
            self.cur_state = self.app_states["START"][0]
            self.init_state = random_choice_from_list(self.app_states["START"])

        self.history_states = []
        self.history_states.append(self.cur_state)

        self.visited_trace = []
        self.visited_trace.append(self.cur_state.img_path)
        self.visited = {k: False for k in self.hash_map.keys()}
        self.visited[self.cur_state.img_path] = True

        self.undefine_op_times = 0
        self.op_times = 0 
        self.score = 0 
        self.is_failed = False
        self._lock = False

    def action(self, act: Action) -> State:
        """
        在 FSM 上执行一个动作，返回新状态。
        """
        if self.cur_state is None:
            # 优先用 START 簇作为起点；否则任选一个簇
            if "START" in self.app_states and self.app_states["START"]:
                self.cur_state = self.app_states["START"][0]
            else:
                any_list = next(iter(self.app_states.values()))
                self.cur_state = random_choice_from_list(any_list)
            self.init_state = self.cur_state

            print(f"random init state {self.cur_state.img_path} {self.cur_state.cluster_class}...")

        # 记录操作历史
        if not self.is_failed:
            print("Cur state img",self.cur_state.img_path,"  Class:",self.cur_state.cluster_class)
            print("\n Cur action ",act)
            self.cur_state = self._transition(act)
            if act.act_type not in ("back","home"):
                self.history_states.append(self.cur_state)
            self.visited_trace.append(self.cur_state.img_path)
            self.visited[self.cur_state.img_path] = True
        if self.op_times > self.max_op_times or self.undefine_op_times > self.max_undefine_op_times:
            self.is_failed = True 
        self.score  = max(self.cur_state.score,self.score)
        return self.cur_state
    
    def get_score(self):
        
        if self.op_times > self.max_trace_step and not self._lock :
            self.score -= max((self.op_times - self.max_trace_step) / (self.max_op_times - self.max_trace_step) * ScoreDeclare.op_times_penalty,0)
        self._lock = True

        return max(self.score,0)



def get_fsm_from_pickle(pickle_path: str) -> AppFSM:
    """
    从 Pickle 文件加载 AppFSM 对象
    """
    import pickle

    with open(pickle_path, 'rb') as f:
        fsm: AppFSM = pickle.load(f)
    return fsm


def build_AppFSM(app: str, task: str, data_path: str) -> AppFSM:
    """
    构建 AppFSM 的便捷函数
    """ 
    data_path = os.path.join(data_path,"rawdata")
    fsm = AppFSM(app, task, data_path, is_init=True,use_cache=False)
    #fsm.save_cache()
    return fsm  

def quick_build_AppFSM(app: str, task: str, data_path: str) -> AppFSM:
    """
    快速构建 AppFSM 的便捷函数（仅解析，不聚簇、不归约）
    """
    data_path = os.path.join(data_path,"rawdata")
    fsm = AppFSM(app, task, data_path, is_init=True,use_cache=True)
    return fsm  


# ----------------------------
# main
# ----------------------------

if __name__ == "__main__":
    with open('/Users/fff/Desktop/mobiagent/MobiBench/data/base.json', 'r', encoding='utf-8') as f:
        alldata = json.load(f)

    datapath = '/Users/fff/Desktop/mobiagent/MobiBench/data'
    for app in alldata.keys():
        for tasktype in alldata[app]:
            #tasklist = get_tasks(app, tasktype)
           
            fsm = build_AppFSM(app=app, task=tasktype, data_path=datapath)
            from MobiBench.utils.score_proc import save_env_result
            save_env_result(
                app=app,
                task=tasktype,
                fsm=fsm,
                savepath=r"/Users/fff/Desktop/mobiagent/MobiBench/runs/dev/env",
            )
            # 让 FSM 内部的 max_op_times 和 CLI 一致
    # fsm.save_traces() 
    # import glob
    # cands = glob.glob(os.path.join(data_path, app, task, "*", "actions.json"))
    # actions_path = sorted(cands)[0] if cands else None

    # if actions_path and os.path.isfile(actions_path):
    #     actions = loadactions(actions_path)
            
            # 让 FSM 内部的 max_op_times 和 CLI 一致
    # fsm.save_traces()
    # import glob
    # cands = glob.glob(os.path.join(data_path, app, task, "*", "actions.json"))
    # actions_path = sorted(cands)[0] if cands else None

    # if actions_path and os.path.isfile(actions_path):
    #     actions = loadactions(actions_path)
    #     fsm.save_traces()  # 输出 fsm_traces.json（与 parsedata.py 输出结构一致）

    #     for a in actions:
    #         st = fsm.action(a)
    #         print(f"action: {a}, state: {st.img_path} {st.cluster_class}")
    #         if getattr(st, "cluster_class", None) == "Done":
    #             print("task done!")
    #             break
    # else:
    #     # 若未指定某条轨迹，至少保存一次解析到的多条轨迹
    #     fsm.save_traces()
    #     print("No specific actions.json for replay; saved aggregated traces instead.")
