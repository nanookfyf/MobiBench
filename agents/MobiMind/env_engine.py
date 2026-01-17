import base64
import io
import json
import logging
import time
import traceback
import numpy as np
#from environments.prompts import GROUNDER_PROMPT
from PIL import Image
#from utils import *
from MobiBench.env.fsm import build_AppFSM,point_in_rectangle
from MobiBench.utils.models.text_match import semantic_similarity
from MobiBench.env.type_spaces import Action
#from agent_system.environments.prompts import GROUNDER_PROMPT


GROUNDER_PROMPT = """
Based on the screenshot, user's intent and the description of the target UI element, provide the bounding box of the element using **absolute coordinates**.
User's intent: {reasoning}
Target element's description: {description}
Your output should be a JSON object with the following format:
{{"bbox": [x1, y1, x2, y2]}}"""


GROUNDER_PROMPT_QWEN3 = '''
Based on user's intent and the description of the target UI element, locate the element in the screenshot.
User's intent: {reasoning}
Target element's description: {description}
Report the bbox coordinates in JSON format.'''

RESIZE_FACTOR = 0.5  # Resize factor for screenshots to reduce size

use_qwen_3 = True

def box2xy(bbox,width,height):
    if use_qwen_3:
        bbox[0] = bbox[0] / 1000 * width
        bbox[2] = bbox[2] / 1000 * width
        bbox[1] = bbox[1] / 1000 * height
        bbox[3] = bbox[3] / 1000 * height
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        x, y = (x1 + x2) // 2, (y1 + y2) // 2
    else:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        x, y = (x1 + x2) // 2, (y1 + y2) // 2
    x, y = int(x / RESIZE_FACTOR), int(y / RESIZE_FACTOR)
    return x, y


class StaticMobiAgentWorker:
    
    def __init__(self,app,task_type,datapath,grounder,use_flag = "e2e_v1",cur_state = None) -> None:
        self.fsm = build_AppFSM(app, task_type, datapath)
        self.fsm._reset()
        self.cur_state = self.fsm.cur_state
        self.grounder_client = grounder # OpenAI(api_key="0", base_url=f"http://{grounder}/v1")
        self.last_state = None   
        self.last_obs = None 
        self.cluster_class_keys = self.fsm.app_states.keys()
        self.use_flag = use_flag


        
    def _get_obs(self):
        img = Image.open(self.cur_state.img_path)
        img = img.resize((int(img.width * RESIZE_FACTOR), int(img.height * RESIZE_FACTOR)), Image.Resampling.LANCZOS)
        self.last_obs = img
        return np.array(img)
    def _get_w_h(self):
        buffer = io.BytesIO()
        self.last_obs.save(buffer, format="JPEG")
        last_obs_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        pil_img = Image.open(io.BytesIO(base64.b64decode(last_obs_base64)))
        width, height = pil_img.size

        return width,height

    def _call_grounder(self, reasoning: str, target_element: str):
        prompt_fmt = GROUNDER_PROMPT if not use_qwen_3 else GROUNDER_PROMPT_QWEN3
        grounder_prompt = prompt_fmt.format(
            reasoning=reasoning,
            description=target_element,
        )
        buffer = io.BytesIO()
        self.last_obs.save(buffer, format="JPEG")
        last_obs_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        pil_img = Image.open(io.BytesIO(base64.b64decode(last_obs_base64)))
        width, height = pil_img.size

        grounder_output = self.grounder_client.chat.completions.create(
            model="",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{last_obs_base64}"}},
                        {"type": "text", "text": grounder_prompt},
                    ]
                }
            ],
            temperature=0
        ).choices[0].message.content
        #print("Grounder prompt:", grounder_prompt)
        if grounder_output.startswith("```json"):
                grounder_output = grounder_output.replace("```json", "").replace("```", "")
        grounder_output_json = json.loads(grounder_output)
        if isinstance(grounder_output_json, list):
                grounder_output_json = grounder_output_json[0]

        print("Grounder response:", grounder_output_json)
        #x1, y1, x2, y2 = grounder_response["bbox"]
        bbox = grounder_output_json.get("bbox", grounder_output_json.get("bbox_2d", None))

        if use_qwen_3:
            bbox[0] = bbox[0] / 1000 * width
            bbox[2] = bbox[2] / 1000 * width
            bbox[1] = bbox[1] / 1000 * height
            bbox[3] = bbox[3] / 1000 * height
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            x, y = (x1 + x2) // 2, (y1 + y2) // 2
        else:
            x1, y1, x2, y2 = grounder_output_json["bbox"]
            x, y = (x1 + x2) // 2, (y1 + y2) // 2
        x, y = int(x / RESIZE_FACTOR), int(y / RESIZE_FACTOR)
        return x, y
    def _handle_click(self,x,y):
        
        for k,v in self.cur_state.map_info["click"].items():
            print("Checking click area:",k)
            if point_in_rectangle(x,y,k[0],k[1],k[2],k[3]):
                self.last_state = self.cur_state
                self.cur_state = self.fsm.hash_map[v]                
                return
            
    def _handle_swipe(self,direction):
        for k,v in self.cur_state.map_info["swipe"].items():
            dir_ = direction.lower()
            if k[0] == dir_:    
                self.last_state = self.cur_state
                self.cur_state =  self.fsm.hash_map[v]
                return 
        
    def _handle_input(self,text):   
        for k,v in self.cur_state.map_info["input"].items():

            if k == text: # match text TODO
                self.last_state = self.cur_state
                self.cur_state =  self.fsm.hash_map[v]
            elif semantic_similarity(k,text)['cosine_similarity']>0.8:
                self.last_state = self.cur_state
                self.cur_state =  self.fsm.hash_map[v]
                return 
    
    def step(self,action):
        
        reward = 0.0
        info = {"status": "ok", "won": 0}
        done = False
        #print("Current image path:", self.cur_state.img_path," Cluster class:", self.cur_state.cluster_class,"\n")
        stdact = None
        try:
            action_type = action["action"]
            parameters = action["parameters"]
            reasoning = action["reasoning"]

            
            if action_type == "click":
                # 使用 Qwen3 模型进行坐标转换
                if self.use_flag == "e2e_v1" or self.use_flag == "e2e_v2":
                    
                    width, height = self._get_w_h()
                    bbox = parameters["bbox"]
                    if bbox is None:
                        logging.error("E2E mode: bbox not found in decider response")
                        raise ValueError("E2E mode requires bbox in decider response")
                    logging.info(f"E2E mode: Using bbox directly from decider: {bbox}")

                    x,y = box2xy(bbox=bbox,width=width,height=height)
                else:

                    target_element = parameters["target_element"]
                    x, y = self._call_grounder(reasoning, target_element)
                stdact = Action(act_type="click",parameters={"position_x":x, "position_y": y})
                self.cur_state = self.fsm.action(stdact)
                
                if self.cur_state.cluster_class=="DONE":
                    done = True
                    info["won"] = 1
                    
            elif action_type == "input":
                text = parameters["text"]
                stdact = Action(act_type="input", parameters={"text": text})
                self.cur_state = self.fsm.action(stdact)
                
                if self.cur_state.cluster_class=="DONE":
                    done = True
                    info["won"] = 1
                
            elif action_type == "swipe":
                # device.swipe(parameters["direction"].lower())
                direction = parameters["direction"]
                stdact = Action(act_type="swipe",parameters={"direction":direction.lower()})
                self.cur_state = self.fsm.action(stdact)
                if self.cur_state.cluster_class=="DONE":
                    done = True
                    info["won"] = 1
                    
            elif action_type == "wait":
                
                stdact = Action(act_type="wait", parameters={})
                self.cur_state = self.fsm.action(stdact)
                if self.cur_state.cluster_class=="DONE":
                    done = True
                    info["won"] = 1
            elif action_type == "click_input":
                text = parameters["text"]
                bbox = parameters["bbox"]
                width, height = self._get_w_h()
                bbox = parameters["bbox"]
                x,y = box2xy(bbox=bbox,width=width,height=height)
                stdact = Action(act_type="click_input", parameters={"position_x":x, "position_y": y,"text":text})
                self.cur_state = self.fsm.action(stdact)
                if self.cur_state.cluster_class=="DONE":
                    done = True
                    info["won"] = 1

            elif action_type == "done":
                stdact = Action(act_type="done", parameters={})
            else:
                logging.info(f"Unknown action type, skipping execution: {action_type}")  
            obs = self._get_obs()

            
            
        except Exception as e:
            reward = -1.0
            done = True
            stdact = None
            obs = None
            info = {"status": "error", "error": traceback.format_exc(), "won": 0}
            logging.error(f"Error during action execution: {e}\n{traceback.format_exc()}")
        

       
        return obs, reward, done, info, self.fsm.is_failed ,stdact 
        
    def reset(self):
        self.cur_state = self.fsm._reset()
        self.cur_state = self.fsm.cur_state
        self.last_state = None
        return self._get_obs(), None
         
    def close(self):
        pass
    