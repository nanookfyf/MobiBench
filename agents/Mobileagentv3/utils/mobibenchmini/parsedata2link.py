"""
根据轨迹构建一个TraceLink对象
"""
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from .type_spaces  import *
import math
from .utils.parse_omni import extract_all_bounds, find_clicked_element
import sys




ActionTypeMAP = {
    
    "click":ClickAction,
    "swipe":SwipeAction,
    "input_text":InputAction,
    "wait":WaitAction,
    "done":DoneAction
            
}

class TraceLink:
    app_name: str
    task_type: str
    task_description: str
    num_states: int
    states: List[State]  # 状态列表
    actions: List[Action] # 动作列表
    def to_dict(self):
        return {
            "app_name": self.app_name,
            "task_type": self.task_type,
            "task_description": self.task_description,
            "num_states": self.num_states,
            "states": [state.to_dict() for state in self.states],
            #"actions": [action.to_dict() for action in self.actions]
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            app_name=data["app_name"],
            task_type=data["task_type"],
            task_description=data["task_description"],
            num_states=data["num_states"],
            states=[State.from_dict(state) for state in data["states"]],
            actions=[Action.from_dict(action) for action in data["actions"]]
        )
    def __len__(self):
        return len(self.states)
        
def save_states(self, states: List[State]):
    """保存状态列表到JSON文件"""
    data = [state.to_dict() for state in states]
    with open(self.filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_states(self) -> List[State]:
    """从JSON文件加载状态列表"""
    try:
        with open(self.filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [State.from_dict(item) for item in data]
    except FileNotFoundError:
        return []
    
class TraceParser:
    """Trace数据解析器"""
    
    def __init__(self):
        self.trace_link = None
        self.states_map = {}  # img_path -> State
    
    def parse_trace_directory(self, trace_dir: str) -> TraceLink:
        """
        解析trace目录，包含actions.json、react.json和截图文件
        
        Args:
            trace_dir: trace数据目录路径
            
        Returns:
            Trace对象
        """
        trace_dir = Path(trace_dir)
        
        # 解析actions.json
        actions_file = trace_dir / "actions.json"
        if not actions_file.exists():
            raise FileNotFoundError(f"actions.json not found in {trace_dir}")
        
        actions = self._load_parse_json_file(actions_file)
        
        # 解析截图文件
        screenshots = self._parse_screenshots(trace_dir)
        
        # 构建Trace对象
        trace = self._build_trace_link(states=screenshots, actions=actions)
        
        return trace
    
    def _load_parse_json_file(self, file_path: Path) -> Dict[str, Any]:
        """加载JSON文件"""
        Actions = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                jsonf = json.load(f)
                self.app_name = jsonf.get("app_name", "unknown")
                self.task_type = jsonf.get("task_type", "unknown")
                self.task_description = jsonf.get("task_description", [])
                self.num_states = jsonf.get("action_count", 0)
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
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {e}")
    
    def _parse_screenshots(self, trace_dir: Path) -> List[State]:
        """解析截图文件"""
        States = []
        
        # 查找所有jpg文件（按数字序排序：1.jpg, 2.jpg, ... 10.jpg）
        jpg_files = sorted(
            trace_dir.glob("*.jpg"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
        )
        
        for i, jpg_file in enumerate(jpg_files):
            screenshot = State(img_path=str(jpg_file))
            self.states_map[str(jpg_file)] = screenshot
            States.append(screenshot)
        
        return States

    def _build_trace_link(self,states,actions):
        self.trace_link = TraceLink()
        self.trace_link.app_name = self.app_name
        self.trace_link.task_type = self.task_type
        self.trace_link.task_description = self.task_description
        self.trace_link.num_states = self.num_states
        self.trace_link.states = states
        self.trace_link.actions = actions
        return self.trace_link

def add_transition(start_state:State, action:Action, target_state:State):
    if action.act_type == "click":
        
        if 'bounds' in action.parameters.keys():
            key = action.parameters['bounds']
            start_state.map_info["click"][tuple(key)] = target_state.img_path
        else:
            if "unknown" not in start_state.map_info["click"]:
                start_state.map_info["click"]["unknown"] = [target_state.img_path]
            else:
                start_state.map_info["click"]["unknown"].append(target_state.img_path)
                
                
    elif action.act_type == "swipe":
        dir_ = action.parameters["direction"]
        dis = 0
        
        if dir_ == "up" or dir_ == "down":
            dis = math.fabs(action.parameters["press_position_y"] - action.parameters["release_position_y"])
            
        elif dir_ == "left" or dir_ == "right":
            dis = math.fabs(action.parameters["press_position_x"] - action.parameters["release_position_x"])
            
        start_state.map_info["swipe"][(dir_,dis)] = target_state.img_path
        
    elif action.act_type == "input":
        start_state.map_info["input"][action.parameters["text"]] = target_state.img_path

    elif action.act_type == "wait":
        start_state.map_info["wait"][action.parameters["duration"]] = target_state.img_path
        
def merge_info(tl):
    actions = getattr(tl, "actions", []) or []
    linkable = [a for a in actions if getattr(a, "act_type", None) in ("click","swipe","input","wait")]

    n_frames = len(tl.states)
    steps = max(0, min(n_frames - 1, len(linkable)))

    # 每帧兜底四类键
    for s in tl.states:
        s.map_info = getattr(s, "map_info", {}) or {}
        for k in ("click","swipe","input","wait"):
            s.map_info.setdefault(k, {})

    for i in range(steps):
        act = linkable[i]; src = tl.states[i]; dst = tl.states[i+1]
        p = getattr(act, "parameters", {}) or {}

        if act.act_type == "click":
            b = p.get("bounds")
            if b and len(b) == 4:
                key = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            else:
                x, y = p.get("position_x"), p.get("position_y")
                if x is not None and y is not None:

                    # 使用视觉模型确定点击位置对应的元素边界框
                    all_bounds = extract_all_bounds(src.img_path)
                    # 找到最小覆盖边框
                    clicked_bounds = find_clicked_element(all_bounds, x, y)
                    if clicked_bounds:
                        key = tuple(clicked_bounds)
                    else:
                        r = 10
                        key = (int(x-r), int(y-r), int(x+r), int(y+r))

                else:
                    #key = "unknown"
                    key = None
                    
            if key:
                src.map_info["click"][key] = dst.img_path

        elif act.act_type == "swipe":
            dir_ = p.get("direction")
            px,py = p.get("press_position_x",0), p.get("press_position_y",0)
            rx,ry = p.get("release_position_x",0), p.get("release_position_y",0)
            dis = abs((py-ry) if dir_ in ("up","down") else (px-rx))
            src.map_info["swipe"][(dir_, int(dis))] = dst.img_path

        elif act.act_type == "input":
            src.map_info["input"][p.get("text","")] = dst.img_path

        else:  # wait
            src.map_info["wait"]["unknown"] = dst.img_path


#test
if __name__ == "__main__":
    parser = TraceParser()
    trace = parser.parse_trace_directory("C:/Users/32089/Desktop/AIinfra/MobiAgent/collect/manual/data/bilibili/type1/4")
    print(f"App Name: {trace.app_name}")
    print(f"Task Type: {trace.task_type}")
    print(f"Task Description: {trace.task_description}")
    print(f"Number of States: {trace.num_states}")
    
    for action in trace.actions:
        print(f"Action Type: {action.act_type}, Parameters: {action.parameters}")
    merge_info(trace)

    for state in trace.states:
        print(f"State Image Path: {state.img_path}, Map Info: {state.map_info}")
    

    
