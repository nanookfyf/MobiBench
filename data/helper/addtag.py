
import json
import os
def rename_images(folder_path, start_id):
    """
    将文件夹中所有 >= start_id 的 jpg 文件编号 +1。
    例如 start_id=3 时，3.jpg -> 4.jpg, 4.jpg -> 5.jpg。
    """

    # 获取所有 jpg 文件名中的数字部分
    jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    # 提取数字编号
    ids = []
    for f in jpg_files:
        try:
            num = int(os.path.splitext(f)[0])
            ids.append(num)
        except ValueError:
            pass  # 忽略非数字命名的文件

    ids.sort(reverse=True)  # 倒序重命名，防止覆盖

    for num in ids:
        if num >= start_id:
            old_name = os.path.join(folder_path, f"{num}.jpg")
            new_name = os.path.join(folder_path, f"{num + 1}.jpg")
            print(f"Renaming {old_name} -> {new_name}")
            os.rename(old_name, new_name)




def insert_tag_action(json_path, insert_index, label_value):
    """
    在指定 JSON 文件的 actions 列表中插入一个新的 tag 动作。

    参数:
        json_path (str): actions.json 文件路径
        insert_index (int): 插入的位置索引（0-based）
        label_value (str): tag 的 label 值，例如 'input1'
    """
    if not os.path.exists(json_path):
        print(f"文件不存在: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "actions" not in data or not isinstance(data["actions"], list):
        print(f"文件格式错误: {json_path}")
        return

    new_action = {
        "type": "tag",
        "label": label_value
    }

    # 限制插入范围
    if insert_index < 0:
        insert_index = 0
    elif insert_index > len(data["actions"]):
        insert_index = len(data["actions"])

    data["actions"].insert(insert_index-1, new_action)

    # 写回原文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"✅ 已在 {json_path} 的第 {insert_index} 个位置插入 label={label_value} 的 tag")

# 示例调用

def help_insert_tag(base_dir,tag_id,tag_label):


    for folder in os.listdir(base_dir):
        sub_path = os.path.join(base_dir, folder)
        if os.path.isdir(sub_path):  # 确保是文件夹
            json_path = os.path.join(sub_path, "actions.json")
            rename_images(sub_path,tag_id)
            insert_tag_action(json_path,tag_id,tag_label)




if __name__ == "__main__":


    folder = r"/Users/fengyunfei/Desktop/mobiagent/MobiBench/data/rawdata/京东/type4"
    #tag_id : 1,2,3...
    help_insert_tag(folder,tag_id=2,tag_label="call_init")