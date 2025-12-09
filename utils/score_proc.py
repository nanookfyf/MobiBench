import os
import pandas as pd
from datetime import datetime
def save_env_result(app, task,fsm, savepath):
    """
    输入：app 任务类型 指令内容 运行fsm
    根据savepath创建表格保存结果，包括得分信息
    """
    # 获取得分
    score = fsm.get_score()
    savepath = os.path.join(savepath,"env_result.csv")
    
    # 创建结果字典
    result_data = {
        'app': [app],
        'task_type': [task],
        'nums of trace link':[len(fsm.traces)],
        'nums of states':[len(fsm.hash_map.keys())],
        'nums of act per state': [[sum(len(v.map_info[act].keys()) for act in ["click","input","swipe","wait"] )for k,v in fsm.hash_map.items()]],
        "nums of total actions":[ sum( sum(len(v.map_info[act].keys()) for act in ["click","input","swipe","wait"] )for k,v in fsm.hash_map.items() ) ],
        "nums of click actions":[  sum(len(v.map_info["click"].keys()) for k,v in fsm.hash_map.items()  ) ],
        "nums of input actions":[ sum( len(v.map_info["input"].keys()) for k,v in fsm.hash_map.items()  ) ],
        "max_step":[fsm.max_trace_step],
        "min_step":[fsm.min_trace_step]
    }
    
    # 转换为DataFrame
    df_new = pd.DataFrame(result_data)
    
    # 检查保存路径是否存在
    if os.path.exists(savepath):
        # 如果文件已存在，读取现有数据并追加新数据
        try:
            df_existing = pd.read_csv(savepath)
            df_combined = pd.concat([df_existing, df_new],ignore_index=True)
            df_combined.to_csv(savepath, index=False)
            print(f"结果已追加到: {savepath}")
        except Exception as e:
            print(f"读取现有文件失败，创建新文件: {e}")
            #df_new.to_csv(savepath, index=False)
    else:
        # 如果文件不存在，创建新文件
        # 确保目录存在
        os.makedirs(os.path.dirname(savepath),  exist_ok=True)
        df_new.to_csv(savepath,index=False)
        print(f"新结果文件已创建: {savepath}")
    
    
def save_result(md, app, task, inst, fsm, time_use,savepath):
    """
    输入：模型名称 app 任务类型 指令内容 运行fsm
    根据savepath创建表格保存结果，包括得分信息
    """
    # 获取得分
    score = fsm.get_score()
    savepath = os.path.join(savepath,f"{md}.csv")
    
    # 创建结果字典
    result_data = {
        'app': [app],
        'task_type': [task],
        'instruction': [inst],
        'score': [round(score, 4)],
        'timeuse':[ round(time_use,4) ],
        'op_times':[ fsm.op_times ],
    }
    
    # 转换为DataFrame
    df_new = pd.DataFrame(result_data)
    
    # 检查保存路径是否存在
    if os.path.exists(savepath):
        # 如果文件已存在，读取现有数据并追加新数据
        try:
            df_existing = pd.read_csv(savepath)
            df_combined = pd.concat([df_existing, df_new],ignore_index=True)
            df_combined.to_csv(savepath, index=False)
            print(f"结果已追加到: {savepath}")
        except Exception as e:
            print(f"读取现有文件失败，创建新文件: {e}")
            df_new.to_csv(savepath, index=False)
    else:
        # 如果文件不存在，创建新文件
        # 确保目录存在
        os.makedirs(os.path.dirname(savepath),  exist_ok=True)
        df_new.to_csv(savepath,index=False)
        print(f"新结果文件已创建: {savepath}")
    
    # 打印结果摘要
    print(f"模型: {md}, 应用: {app}, 任务: {task}, 得分: {score}")

# 使用示例
# save_result('Mobile-Agent-v2', '微信', '发送消息', '给张三发送你好', fsm_instance, './results/evaluation_results.csv')

def dict2csv(data,path):

    import pandas as pd
    import os


    df = pd.DataFrame(data)

    # 追加到CSV文件
    csv_file = path
    if os.path.exists(csv_file):
        # 如果文件存在，追加模式，不写入列名
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        # 如果文件不存在，新建文件，写入列名
        df.to_csv(csv_file, mode='w', header=True, index=False)