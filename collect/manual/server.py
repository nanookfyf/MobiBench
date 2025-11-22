from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import json
import base64
import shutil
import uvicorn
import uiautomator2 as u2
import sys
import os
from typing import Optional, Dict, Any, Union
from utils.parse_xml import find_clicked_element

# 数据模型
class ClickAction(BaseModel):
    x: int
    y: int

class SwipeAction(BaseModel):
    startX: int
    startY: int
    endX: int
    endY: int
    direction: str  # 'up', 'down', 'left', 'right'

class InputAction(BaseModel):
    text: str

class TaskDescription(BaseModel):
    description: str
    app_name: str
    task_type: str

class AnnotateAction(BaseModel):
    label: Union[int, str]

screenshot_path = "screenshot.jpg"

currentDataIndex = 0
action_history = []
current_task_description = ""  # 当前任务描述
current_app_name = ""  # 当前应用名称
current_task_type = ""  # 当前任务类型

device = None  # 设备连接对象
hierarchy = None  # 层次结构数据

app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件服务
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

def save_screenshot():
    action_count = len(action_history)

    # 创建数据目录
    session_base_dir = os.path.dirname(__file__)
    data_base_dir = os.path.join(session_base_dir, 'data')
    app_dir = os.path.join(data_base_dir, current_app_name)
    task_type_dir = os.path.join(app_dir, current_task_type)
    data_dir = os.path.join(task_type_dir, str(currentDataIndex))

    # 复制当前截图到数据目录
    if os.path.exists(screenshot_path):
        screenshot_save_path = os.path.join(data_dir, f'{action_count + 1}.jpg')
        shutil.copy2(screenshot_path, screenshot_save_path)

def get_current_hierarchy_and_screenshot(sleep_time = 0):
    global hierarchy
    time.sleep(sleep_time)
    hierarchy = device.dump_hierarchy()
    
    # with open("hierarchy.xml", "w", encoding="utf-8") as f:
    #     f.write(hierarchy)

    device.screenshot(screenshot_path)
    print(f"操作完成，已重新截图和获取层次结构。总操作数: {len(action_history)}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """返回前端页面"""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/screenshot")
async def get_screenshot():
    """获取最新截图文件和层次结构信息"""
    try:
        get_current_hierarchy_and_screenshot()
        with open(screenshot_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
        return {
            "status": "success",
            "image_data": f"data:image/jpeg;base64,{image_data}",
            "hierarchy": hierarchy,
            "timestamp": int(time.time() * 1000)
        }
      
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取截图失败: {str(e)}")

@app.post("/click")
async def handle_click(action: ClickAction):
    """处理点击操作"""
    try:
        # 确保坐标为整数（舍入）
        x = round(action.x)
        y = round(action.y)
        
        element_bounds = find_clicked_element(hierarchy, x, y)
        if element_bounds:
            element_bounds = [round(coord) for coord in element_bounds]
        
        get_current_hierarchy_and_screenshot()
        save_screenshot()
        device.click(x, y)
        action_record = {
            "type": "click",
            "position_x": x,
            "position_y": y,
            "bounds": element_bounds,
        }
        print(action_record)
        action_history.append(action_record)
        # get_current_hierarchy_and_screenshot(1.5)

        return {
            "status": "success", 
            "message": f"点击操作完成: ({x}, {y})",
            "action": "click",
            "coordinates": {"x": x, "y": y},
            "clicked_bounds": element_bounds,
            "action_count": len(action_history)
        }
    
    except Exception as e:
        print(f"点击操作失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"点击操作失败: {str(e)}")

@app.post("/swipe")
async def handle_swipe(action: SwipeAction):
    """处理滑动操作"""
    try:
        # 确保坐标为整数（舍入）
        startX = round(action.startX)
        startY = round(action.startY)
        endX = round(action.endX)
        endY = round(action.endY)
        
        get_current_hierarchy_and_screenshot()
        save_screenshot()
        device.swipe(startX, startY, endX, endY, duration=0.1)
        action_record = {
            "type": "swipe",
            "press_position_x": startX,
            "press_position_y": startY,
            "release_position_x": endX,
            "release_position_y": endY,
            "direction": action.direction,
        }
        print(action_record)
        action_history.append(action_record)
        # get_current_hierarchy_and_screenshot(1.5)

        return {
            "status": "success",
            "message": f"滑动操作完成: ({startX}, {startY}) → ({endX}, {endY}) [{action.direction}]",
            "action": "swipe",
            "start": {"x": startX, "y": startY},
            "end": {"x": endX, "y": endY},
            "direction": action.direction,
            "action_count": len(action_history)
        }
    
    except Exception as e:
        print(f"滑动操作失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"滑动操作失败: {str(e)}")

@app.post("/input")
async def handle_input(action: InputAction):
    try:
        get_current_hierarchy_and_screenshot()
        save_screenshot()
        current_ime = device.current_ime()
        device.shell(['settings', 'put', 'secure', 'default_input_method', 'com.android.adbkeyboard/.AdbIME'])
        time.sleep(0.5)
        charsb64 = base64.b64encode(action.text.encode('utf-8')).decode('utf-8')
        device.shell(['am', 'broadcast', '-a', 'ADB_INPUT_B64', '--es', 'msg', charsb64])
        time.sleep(0.5)
        device.shell(['settings', 'put', 'secure', 'default_input_method', current_ime])
        action_record = {
            "type": "input",
            "text": action.text,
        }
        print(action_record)
        action_history.append(action_record)
        # get_current_hierarchy_and_screenshot(1.5)

        return {
            "status": "success",
            "message": f"输入操作完成",
            "action": "input",
            "text": action.text,
            "action_count": len(action_history)
        }
    
    except Exception as e:
        print(f"输入操作失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"输入操作失败: {str(e)}")

@app.get("/action_history")
async def get_action_history():
    """获取操作历史记录"""
    return {
        "status": "success",
        "total_actions": len(action_history),
        "actions": action_history
    }

@app.post("/save_data")
async def save_current_data():
    """保存当前数据并清空历史记录"""
    global currentDataIndex
    global action_history

    try:
        get_current_hierarchy_and_screenshot()
        save_screenshot()
        action_record = {
            "type": "done"
        }
        action_history.append(action_record)
        action_count = len(action_history)

        app_dir = os.path.join(os.path.dirname(__file__), 'data', current_app_name)
        task_type_dir = os.path.join(app_dir, current_task_type)
        data_dir = os.path.join(task_type_dir, str(currentDataIndex))
        json_file_path = os.path.join(data_dir, 'actions.json')
        
        save_data = {
            "app_name": current_app_name,
            "task_type": current_task_type,
            "task_description": current_task_description,
            "action_count": action_count,
            "actions": action_history
        }
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=4)
  
        action_history.clear()

        # [Info]
        print(f"第 {currentDataIndex} 条数据已保存")
        print(f"应用：{current_app_name} | 任务类型：{current_task_type}")
        print(f"包含 {action_count} 个操作记录")
        print("操作历史记录已清空")
        
        return {
            "status": "success",
            "message": f"第 {currentDataIndex} 条数据已保存",
            "data_index": currentDataIndex,
            "saved_actions": action_count
        }
    except Exception as e:
        print(f"保存数据失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"保存数据失败: {str(e)}")

@app.post("/delete_data")
async def delete_current_data():
    """保存当前数据并清空历史记录"""
    global currentDataIndex

    try:
        app_dir = os.path.join(os.path.dirname(__file__), 'data', current_app_name)
        task_type_dir = os.path.join(app_dir, current_task_type)
        data_dir = os.path.join(task_type_dir, str(currentDataIndex))

        # 删除数据目录
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
    
        action_history.clear()

        return {
            "status": "success",
            "message": f"第 {currentDataIndex} 条数据已删除",
            "data_index": currentDataIndex
        }
    except Exception as e:
        print(f"保存数据失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"保存数据失败: {str(e)}")
    

@app.post("/annotate")
async def annotate(action: AnnotateAction):
    try:
        # 与其他动作一致：先抓取最新层级与截图（如果你项目中就是这样做）
        get_current_hierarchy_and_screenshot()

        # 顺带把动作记入内存历史（便于你回放或生成状态图）
        action_record = {
            "type": "tag",
            "label": action.label,
        }
        action_history.append(action_record)

        return {
            "status": "success",
            "action": "tag",
            "label": action.label,
            "action_count": len(action_history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"标注失败: {str(e)}")


app_packages ={
    "微信": "com.tencent.mm",
    "QQ": "com.tencent.mobileqq",
    "微博": "com.sina.weibo",
    
    "饿了么": "me.ele",
    "美团": "com.sankuai.meituan",

    "bilibili": "tv.danmaku.bili",
    "爱奇艺": "com.qiyi.video",
    "腾讯视频": "com.tencent.qqlive",
    "优酷": "com.youku.phone",

    "淘宝": "com.taobao.taobao",
    "京东": "com.jingdong.app.mall",

    "携程": "ctrip.android.view",
    "同城": "com.tongcheng.android",
    "飞猪": "com.taobao.trip",
    "去哪儿": "com.Qunar",
    "华住会": "com.htinns",

    "知乎": "com.zhihu.android",
    "小红书": "com.xingin.xhs",

    "QQ音乐": "com.tencent.qqmusic",
    "网易云音乐": "com.netease.cloudmusic",
    "酷狗音乐": "com.kugou.android",

    "高德地图": "com.autonavi.minimap"
}

@app.post("/set_task_description")
async def set_task_description(task: TaskDescription):
    """设置任务描述"""
    global currentDataIndex
    global current_task_description
    global current_app_name
    global current_task_type
    try:
        current_app_name = task.app_name
        current_task_type = task.task_type
        current_task_description = task.description

        # 创建新的目录结构：data/<应用名称>/<任务类型>/<数据索引>/
        session_base_dir = os.path.dirname(__file__)
        if not os.path.exists(session_base_dir):
            os.makedirs(session_base_dir)

        data_base_dir = os.path.join(session_base_dir, 'data')
        if not os.path.exists(data_base_dir):
            os.makedirs(data_base_dir)
        
        app_dir = os.path.join(data_base_dir, current_app_name)
        if not os.path.exists(app_dir):
            os.makedirs(app_dir)
            
        task_type_dir = os.path.join(app_dir, current_task_type)
        if not os.path.exists(task_type_dir):
            os.makedirs(task_type_dir)

        # 遍历现有数据目录，找到最大的索引
        existing_dirs = [d for d in os.listdir(task_type_dir) if os.path.isdir(os.path.join(task_type_dir, d)) and d.isdigit()]
        if existing_dirs:
            currentDataIndex = max(int(d) for d in existing_dirs) + 1
        else:
            currentDataIndex = 1
        data_dir = os.path.join(task_type_dir, str(currentDataIndex))
        os.makedirs(data_dir)

        print(f"\n{'='*50}")
        print(f"📋 新任务开始")
        print(f"应用名称: {current_app_name}")
        print(f"任务类型: {current_task_type}")
        print(f"任务描述: {current_task_description}")
        print(f"数据目录: data/{current_app_name}/{current_task_type}/{currentDataIndex}/")
        print(f"{'='*50}\n")
        
        package_name = app_packages.get(current_app_name)
        if not package_name:
            raise ValueError(f"App '{app}' is not registered with a package name.")
        device.app_start(package_name, stop=True)

        return {
            "status": "success", 
            "message": "任务描述已设置",
            "description": current_task_description,
            "app_name": current_app_name,
            "task_type": current_task_type
        }
    except Exception as e:
        print(f"设置任务描述失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"设置任务描述失败: {str(e)}")

if __name__ == "__main__":
    device = u2.connect()
    print("启动服务器...")
    print("访问 http://localhost:9000 查看前端页面")
    uvicorn.run(app, port=9000)