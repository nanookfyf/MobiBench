import json
import os
def get_tasks(app,tasktype):

    current_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件夹
    parent_path = os.path.dirname(current_path)   # 父文件夹
    path = os.path.join(parent_path, 'data',"rawdata", app,tasktype,'task.json')
    with open(path,'r',encoding='utf-8') as f:
        task_data = json.load(f)
    tasks = task_data
    return tasks

def get_tasks_1(app,tasktype):

    current_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件夹
    parent_path = os.path.dirname(current_path)   # 父文件夹
    path = os.path.join(parent_path, 'data',"rawdata", app,tasktype,'task1.json')
    with open(path,'r',encoding='utf-8') as f:
        task_data = json.load(f)
    tasks = task_data
    return tasks