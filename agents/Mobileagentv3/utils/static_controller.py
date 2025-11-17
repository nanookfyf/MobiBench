import os
import time
import subprocess
from .controller import Controller
from MobiBench.env.type_spaces import Action
from PIL import Image

class StaticController(Controller):

    def __init__(self, fsm):
        self.fsm = fsm
        self.fsm._reset()
        self.cur_state = self.fsm.cur_state

    def get_screenshot(self, save_path):
        img = Image.open(self.cur_state.img_path)
        img.save(save_path)

        if not os.path.exists(save_path):
            return False
        else:
            return True

    def tap(self, x, y):
        std_action = Action(act_type="click", parameters={"position_x": x, "position_y": y})
        self.cur_state = self.fsm.action(std_action)
        

    def slide(self, x1, y1, x2, y2):
        std_action = Action(act_type="swipe", parameters={"start_x": x1, "start_y": y1, "end_x": x2, "end_y": y2})
        self.cur_state = self.fsm.action(std_action)
    
    def type(self, text):
        std_action = Action(act_type="input", parameters={"text": text})
        self.cur_state = self.fsm.action(std_action)

    def back(self):
        std_action = Action(act_type="back", parameters={})
        self.cur_state = self.fsm.action(std_action)

    def home(self):
        std_action = Action(act_type="home", parameters={})
        self.cur_state = self.fsm.action(std_action)

    def reset(self):
        self.fsm._reset()
        self.cur_state = self.fsm.cur_state



