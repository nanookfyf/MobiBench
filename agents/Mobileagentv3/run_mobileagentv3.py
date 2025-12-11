import os
import uuid
import json
import time
import argparse
from PIL import Image
from datetime import datetime
from MobiBench.env.fsm import build_AppFSM
from MobiBench.agents.Mobileagentv3.utils.mobile_agent_e import (
    InfoPool, 
    Manager, 
    Executor, 
    Notetaker, 
    ActionReflector,
    INPUT_KNOW
)
import MobiBench.agents.Mobileagentv3.utils.controller as controller
from MobiBench.agents.Mobileagentv3.utils.call_mobile_agent_e import GUIOwlWrapper
from MobiBench.utils.score_proc import save_result,dict2csv
def run_instruction(ctl,api_key,base_url, model, instruction, add_info, coor_type, if_notetaker, max_step=25, log_path="/Users/fengyunfei/Desktop/mobiagent/MobiBench/agents/Mobileagentv3/logs"):
    controller = ctl 
    now = datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")
    save_path = f"{log_path}/{time_str}_{instruction[:10]}"
    os.mkdir(save_path)
    image_save_path = os.path.join(save_path, "images")
    os.mkdir(image_save_path)

    info_pool = InfoPool(
        additional_knowledge_manager=add_info,
        additional_knowledge_executor=INPUT_KNOW,
        err_to_manager_thresh=2
    )
    
    vllm = GUIOwlWrapper(api_key, base_url, model)
    manager = Manager()
    executor = Executor()
    notetaker = Notetaker()
    action_reflector = ActionReflector()
    message_manager, message_operator, message_reflector, message_notekeeper = None, None, None, None
    info_pool.instruction = instruction

    step = 1
    manage_num_toks = 0
    operator_num_toks = 0
    reflector_num_toks = 0

    manage_num_p_toks = 0
    operator_num_p_toks = 0
    reflector_num_p_toks = 0

    manage_num_d_toks = 0
    operator_num_d_toks = 0
    reflector_num_d_toks = 0
    while step <= controller.fsm.max_op_times:

        if step == max_step:
            task_result_path = os.path.join(save_path, "task_result.json")
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
            task_result_data = {"goal": instruction, "finish_dtime": formatted_time, "hit_step_limit": 1.0}
            with open(task_result_path, 'w', encoding='utf-8') as json_file:
                json.dump(task_result_data, json_file, ensure_ascii=False, indent=4)
            break
        
        if step == 1:
            current_time = datetime.now()
            formatted_time = current_time.strftime(f'%Y-%m-%d-{current_time.hour * 3600 + current_time.minute * 60 + current_time.second}-{str(uuid.uuid4().hex[:8])}')
            local_image_dir = os.path.join(image_save_path, f"screenshot_{formatted_time}.png")
        else:
            local_image_dir = local_image_dir2
        
        # get the screenshot
        for _ in range(5):
            if not controller.get_screenshot(local_image_dir):
                print("Get screenshot failed, retry.")
                time.sleep(5)
            else:
                break
        
        width, height = Image.open(local_image_dir).size
        
        info_pool.error_flag_plan = False
        err_to_manager_thresh = info_pool.err_to_manager_thresh
        if len(info_pool.action_outcomes) >= err_to_manager_thresh:
            # check if the last err_to_manager_thresh actions are all errors
            latest_outcomes = info_pool.action_outcomes[-err_to_manager_thresh:]
            count = 0
            for outcome in latest_outcomes:
                if outcome in ["B", "C"]:
                    count += 1
            if count == err_to_manager_thresh:
                info_pool.error_flag_plan = True

        skip_manager = False
        ## if previous action is invalid, skip the manager and try again first ##
        if not info_pool.error_flag_plan and len(info_pool.action_history) > 0:
            if info_pool.action_history[-1]['action'] == 'invalid':
                skip_manager = True
        
        if not skip_manager:
            print("\n### Manager ... ###\n")
            prompt_planning = manager.get_prompt(info_pool)
            output_planning, message_manager, raw_response = vllm.predict_mm(
                prompt_planning,
                [local_image_dir]
            )
            manage_num_toks += raw_response.usage.total_tokens
            manage_num_p_toks += raw_response.usage.prompt_tokens
            manage_num_d_toks += raw_response.usage.completion_tokens
        
        message_save_path = os.path.join(save_path, f"step_{step+1}")
        os.mkdir(message_save_path)
        message_file = os.path.join(message_save_path, "manager.json")
        message_data = {"name": "manager", "messages": message_manager, "response": output_planning, "step_id": step+1}
        with open(message_file, 'w', encoding='utf-8') as json_file:
            json.dump(message_data, json_file, ensure_ascii=False, indent=4)

        parsed_result_planning = manager.parse_response(output_planning)
        info_pool.completed_plan = parsed_result_planning['completed_subgoal']
        info_pool.plan = parsed_result_planning['plan']
        if not raw_response:
            raise RuntimeError('Error calling vLLM in planning phase.')
        
        print('Completed subgoal: ' + info_pool.completed_plan)
        print('Planning thought: ' + parsed_result_planning['thought'])
        print('Plan: ' + info_pool.plan, "\n")
        if "Finished" in info_pool.plan.strip() : #公平起见去除最少操作次数
            print("Plan Model think finished !!")
            break

        if controller.fsm.is_failed:
            print("Plan Model think finished !!")
            break


        if controller.cur_state.cluster_class=="DONE":
            print("Instruction finished, stop the process.")
            task_result_path = os.path.join(save_path, "task_result.json")
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
            task_result_data = {"goal": instruction, "finish_dtime": formatted_time, "hit_step_limit": 0.0}
            with open(task_result_path, 'w', encoding='utf-8') as json_file:
                json.dump(task_result_data, json_file, ensure_ascii=False, indent=4)
            break
        else:
            print("\n### Operator ... ###\n")

            prompt_action = executor.get_prompt(info_pool)
            output_action, message_operator, raw_response = vllm.predict_mm(
                prompt_action,
                [local_image_dir],
            )
            
            #operator_num_toks += raw_response.usage.total_tokens
            operator_num_toks += raw_response.usage.total_tokens
            operator_num_p_toks += raw_response.usage.prompt_tokens
            operator_num_d_toks += raw_response.usage.completion_tokens

            if not raw_response:
                raise RuntimeError('Error calling LLM in operator phase.')
            parsed_result_action = executor.parse_response(output_action)
            action_thought, action_object_str, action_description = parsed_result_action['thought'], parsed_result_action['action'], parsed_result_action['description']
            
            info_pool.last_action_thought = action_thought
            info_pool.last_summary = action_description
            
            if (not action_thought) or (not action_object_str):
                print('Action prompt output is not in the correct format.')
                info_pool.last_action = {"action": "invalid"}
                info_pool.action_history.append({"action": "invalid"})
                info_pool.summary_history.append(action_description)
                info_pool.action_outcomes.append("C")
                info_pool.error_descriptions.append("invalid action format, do nothing.")
                continue
        
        action_object_str = action_object_str.replace("```", "").replace("json", "").strip()
        print('Thought: ' + action_thought)
        print('Action: ' + action_object_str)
        print('Action description: ' + action_description)

        try:
            action_object = json.loads(action_object_str)
            operator_response = f'''### Thought ###
                {action_thought}

                ### Action ###
                {action_object}

                ### Description ###
                {action_description}
                '''
            
            if action_object['action'] == "answer":
                message_file = os.path.join(message_save_path, "operator.json")
                message_data = {"name": "operator", "messages": message_operator, "response": operator_response, "step_id": step+1}
                with open(message_file, 'w', encoding='utf-8') as json_file:
                    json.dump(message_data, json_file, ensure_ascii=False, indent=4)

                answer_content = action_object['text']
                print(f"Instruction finished, answer: {answer_content}, stop the process.")
                task_result_path = os.path.join(save_path, "task_result.json")
                current_time = datetime.now()
                formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
                task_result_data = {"goal": instruction, "finish_dtime": formatted_time, "hit_step_limit": 0.0}
                with open(task_result_path, 'w', encoding='utf-8') as json_file:
                    json.dump(task_result_data, json_file, ensure_ascii=False, indent=4)
                break
            
            if coor_type != "abs":
                if "coordinate" in action_object:
                    action_object['coordinate'] = [int(action_object['coordinate'][0] / 1000 * width), int(action_object['coordinate'][1] / 1000 * height)]
                if "coordinate2" in action_object:
                    action_object['coordinate2'] = [int(action_object['coordinate2'][0] / 1000 * width), int(action_object['coordinate2'][1] / 1000 * height)]
            
            if action_object['action'] == "click":
                controller.tap(action_object['coordinate'][0], action_object['coordinate'][1])
            elif action_object['action'] == "swipe":
                controller.slide(action_object['coordinate'][0], action_object['coordinate'][1], action_object['coordinate2'][0], action_object['coordinate2'][1])
            elif action_object['action'] == "type":
                controller.type(action_object['text'])
            elif action_object['action'] == "system_button":
                if action_object['button'] == "Back":
                    controller.back()
                elif action_object['button'] == "Home":
                    controller.home()
            
        except:
            info_pool.last_action = {"action": "invalid"}
            info_pool.action_history.append({"action": "invalid"})
            info_pool.summary_history.append(action_description)
            info_pool.action_outcomes.append("C")
            info_pool.error_descriptions.append("invalid action format, do nothing.")
            local_image_dir2 = local_image_dir
            continue
        
        message_file = os.path.join(message_save_path, "operator.json")
        message_data = {"name": "operator", "messages": message_operator, "response": operator_response, "step_id": step+1}
        with open(message_file, 'w', encoding='utf-8') as json_file:
            json.dump(message_data, json_file, ensure_ascii=False, indent=4)

        info_pool.last_action = json.loads(action_object_str)
        
        if step == 1:
            time.sleep(8) # maybe a pop-up when first open an app
        time.sleep(2)
        
        current_time = datetime.now()
        formatted_time = current_time.strftime(f'%Y-%m-%d-{current_time.hour * 3600 + current_time.minute * 60 + current_time.second}-{str(uuid.uuid4().hex[:8])}')
        local_image_dir2 = os.path.join(image_save_path, f"screenshot_{formatted_time}.png")
        
        # get the screenshot
        for _ in range(5):
            if not controller.get_screenshot(local_image_dir2):
                print("Get screenshot failed, retry.")
                time.sleep(5)
            else:
                break
        
        print("\n### Action Reflector ... ###\n")
        prompt_action_reflect = action_reflector.get_prompt(info_pool)
        output_action_reflect, message_reflector, raw_response = vllm.predict_mm(
            prompt_action_reflect,
            [
                local_image_dir,
                local_image_dir2,
            ],
        )
        #reflector_num_toks += raw_response.usage.total_tokens
        reflector_num_toks += raw_response.usage.total_tokens
        reflector_num_p_toks += raw_response.usage.prompt_tokens
        reflector_num_d_toks += raw_response.usage.completion_tokens
    
        message_file = os.path.join(message_save_path, "reflector.json")
        message_data = {"name": "reflector", "messages": message_reflector, "response": output_action_reflect, "step_id": step+1}
        with open(message_file, 'w', encoding='utf-8') as json_file:
            json.dump(message_data, json_file, ensure_ascii=False, indent=4)
        
        parsed_result_action_reflect = action_reflector.parse_response(output_action_reflect)
        outcome, error_description = (
            parsed_result_action_reflect['outcome'], 
            parsed_result_action_reflect['error_description']
        )
        progress_status = info_pool.completed_plan
        
        if "A" in outcome: # Successful. The result of the last action meets the expectation.
          action_outcome = "A"
        elif "B" in outcome: # Failed. The last action results in a wrong page. I need to return to the previous state.
            action_outcome = "B"
        elif "C" in outcome: # Failed. The last action produces no changes.
            action_outcome = "C"
        else:
            raise ValueError("Invalid outcome:", outcome)
        
        print('Action reflection outcome: ' + action_outcome)
        print('Action reflection error description: ' + error_description)
        print('Action reflection progress status: ' + progress_status, "\n")
        
        info_pool.action_history.append(json.loads(action_object_str))
        info_pool.summary_history.append(action_description)
        info_pool.action_outcomes.append(action_outcome)
        info_pool.error_descriptions.append(error_description)
        info_pool.progress_status = progress_status
        
        if action_outcome == "A" and if_notetaker:
            print("\n### NoteKeeper ... ###\n")
            prompt_note = notetaker.get_prompt(info_pool)
            output_note, message_notekeeper, raw_response = vllm.predict_mm(
                prompt_note,
                [local_image_dir2],
            )
            
            message_file = os.path.join(message_save_path, "notekeeper.json")
            message_data = {"name": "notekeeper", "messages": message_notekeeper, "response": output_note, "step_id": step+1}
            with open(message_file, 'w', encoding='utf-8') as json_file:
                json.dump(message_data, json_file, ensure_ascii=False, indent=4)
            
            parsed_result_note = notetaker.parse_response(output_note)
            important_notes = parsed_result_note['important_notes']
            info_pool.important_notes = important_notes

            print('Important notes: ' + important_notes, "\n")
        step += 1
    print(f"avg manager toks{manage_num_toks/step} | avg operator toks {operator_num_toks/step} | avg reflector toks {reflector_num_toks/step}")
    dict_data = {
        "avg_manager_toks":[manage_num_toks/step],
        "avg_manager_p_toks":[manage_num_p_toks/step],
        "avg_manager_d_toks":[manage_num_d_toks/step],
        "avg_operator_toks":[operator_num_toks/step],
        "avg_operator_p_toks":[operator_num_p_toks/step],
        "avg_operator_d_toks":[operator_num_d_toks/step],
        "avg_reflector_toks":[reflector_num_toks/step],
        "avg_reflector_p_toks":[reflector_num_p_toks/step],
        "avg_reflector_d_toks":[reflector_num_d_toks/step]

    }
    dict2csv(dict_data,"/Users/fengyunfei/Desktop/mobiagent/MobiBench/results/dev/v3toks.csv")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run Mobile-Agent-v3 with a given model and instruction"
    )
    parser.add_argument("--adb_path", type=str)
    parser.add_argument("--hdc_path", type=str)   
    parser.add_argument("--api_key", type=str,default="")
    parser.add_argument("--base_url", type=str,default="http://123.60.91.241:9000/v1")
    parser.add_argument("--model", type=str,default="")
    parser.add_argument("--instruction", type=str)
    parser.add_argument("--add_info", type=str, default="")
    parser.add_argument("--coor_type", type=str, default="abs")
    parser.add_argument("--datapath", type=str, default="/Users/fengyunfei/Desktop/mobiagent/MobiBench/data", help="path to data")
    parser.add_argument("--notetaker", type=bool, default=False)
    args = parser.parse_args()
    
    from MobiBench.agents.Mobileagentv3.utils.static_controller import StaticController
    from MobiBench.env.fsm import build_AppFSM
    import logging
    from MobiBench.utils.task_get import get_tasks,get_tasks_1
    import time
    #app_list = [ "高德","京东", "美团","淘宝","网易云音乐","微博","小红书"]
    #type_list = ["type8","type9","type10"]
    #type_list = ["type4"]
    type_list = ["type1","type2"]
    app_list = [ "QQ"]
    type_list_hash = {
        "微博" : ["type3"]
    }
    #app_list = [ "同城"]
    #type_list = ["type5"]
    with open('/Users/fengyunfei/Desktop/mobiagent/MobiBench/data/follow1.json', 'r', encoding='utf-8') as f:
        alldata = json.load(f)

    datapath = args.datapath
    for app in alldata.keys():
        for tasktype in alldata[app]:
            tasklist = get_tasks(app,tasktype)
            fsm = build_AppFSM(app=app,task=tasktype,data_path=datapath)
            ctl = StaticController(fsm=fsm)
            for task in tasklist:
                ctl.reset()
                print(f"Handle : {app} : {tasktype} : {task}")
                start = time.time()
                run_instruction(ctl, args.api_key,args.base_url, args.model, task, args.add_info, args.coor_type, args.notetaker)
                print("finished state",fsm.cur_state.img_path)
                end = time.time()
                
                save_result(md="MobiAgentv3",app=app,task=tasktype,inst=task,fsm=ctl.fsm,time_use=end-start,savepath="/Users/fengyunfei/Desktop/mobiagent/MobiBench/results/dev")
 

                

           
                
           
    