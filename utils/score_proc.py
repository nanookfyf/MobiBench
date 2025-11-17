import os
import pandas as pd
from datetime import datetime

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
        'score': [score],
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