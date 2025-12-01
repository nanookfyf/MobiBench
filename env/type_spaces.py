
"""定义状态空间和动作空间 """

from enum import Enum
from dataclasses import dataclass,field,asdict
from typing import Callable, Dict, Optional

@dataclass
class State:
    '''
        app_state 
    '''
    img_path : str = None# 界面截图路径
    
    map_info : Dict[str, Dict] = field(default_factory=lambda: {
        "click": {},
        "swipe": {},
        "input": {},
        "wait": {}
    })
    
    cluster_class : str = None # 状态簇类别
    score : float = 0.0  # 状态评分
    
    def to_dict(self):
        data = asdict(self)
        # 清理数据中的元组键
        return self._clean_data(data)
    
    def _clean_data(self, data):
        """递归清理数据，将元组键转换为字符串"""
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                if isinstance(key, tuple):
                    new_key = f"TUPLE:{key}"
                else:
                    new_key = key
                cleaned[new_key] = self._clean_data(value)
            return cleaned
        elif isinstance(data, list):
            return [self._clean_data(item) for item in data]
        else:
            return data
    
    @classmethod
    def from_dict(cls, data):
        # 恢复元组键
        data = cls._restore_data(data)
        return cls(**data)
    
    @classmethod
    def _restore_data(cls, data):
        """递归恢复数据中的元组键"""
        if isinstance(data, dict):
            restored = {}
            for key, value in data.items():
                if isinstance(key, str) and key.startswith("TUPLE:"):
                    try:
                        new_key = eval(key[6:])
                    except:
                        new_key = key
                else:
                    new_key = key
                restored[new_key] = cls._restore_data(value)
            return restored
        elif isinstance(data, list):
            return [cls._restore_data(item) for item in data]
        else:
            return data
    
   
@dataclass
class Action:
    '''
        app action
    '''
    act_type : str
    parameters : dict
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

# action detail
class ClickAction:
    x: int
    y: int
class SwipeAction:
    startX: int
    startY: int
    endX: int
    endY: int
    direction: str  # 'up', 'down', 'left', 'right'
class InputAction:
    text: str
class WaitAction:
    duration: int  # 等待时间，单位可自定义(s,ms,us)
class DoneAction:
    pass 
class ActionSpace:
    click : ClickAction # 点击坐标 and 组件区域
    swipe : SwipeAction # 滑动起始坐标 and 终止坐标
    input_text : InputAction # 输入文本
    wait : WaitAction # 等待时间 ，单位可自定义(s,ms,us)