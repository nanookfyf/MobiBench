"""System prompts for the AI agent."""
import json
from datetime import datetime
from typing import Any, Callable
from PIL import Image
import base64
import time
from dataclasses import dataclass, field
from openai import OpenAI
import traceback
import ast
from MobiBench.utils.draw_bounds import process_folder
from MobiBench.utils.task_get import get_tasks,get_tasks_1
from MobiBench.env.fsm import build_AppFSM, quick_build_AppFSM
from datetime import datetime
from MobiBench.env.type_spaces import Action
@dataclass
class ModelConfig:
    """Configuration for the AI model."""

    base_url: str = f"http://123.60.91.241:9003/v1"
    api_key: str = "0"
    model_name: str = ""
    max_tokens: int = 30000
    temperature: float = 0.0
    top_p: float = 0.85
    frequency_penalty: float = 0.2
    extra_body: dict[str, Any] = field(default_factory=dict)
    lang: str = "cn"  # Language for UI messages: 'cn' or 'en'


@dataclass
class ModelResponse:
    """Response from the AI model."""

    thinking: str
    action: str
    raw_content: str
    # Performance metrics
    time_to_first_token: float | None = None  # Time to first token (seconds)
    time_to_thinking_end: float | None = None  # Time to thinking end (seconds)
    total_time: float | None = None  # Total inference time (seconds)

@dataclass
class StepResult:
    """Result of a single agent step."""

    success: bool
    finished: bool
    action: dict[str, Any] | None
    thinking: str
    message: str | None = None


"""Internationalization (i18n) module for Phone Agent UI messages."""

# Chinese messages
MESSAGES_ZH = {
    "thinking": "思考过程",
    "action": "执行动作",
    "task_completed": "任务完成",
    "done": "完成",
    "starting_task": "开始执行任务",
    "final_result": "最终结果",
    "task_result": "任务结果",
    "confirmation_required": "需要确认",
    "continue_prompt": "是否继续？(y/n)",
    "manual_operation_required": "需要人工操作",
    "manual_operation_hint": "请手动完成操作...",
    "press_enter_when_done": "完成后按回车继续",
    "connection_failed": "连接失败",
    "connection_successful": "连接成功",
    "step": "步骤",
    "task": "任务",
    "result": "结果",
    "performance_metrics": "性能指标",
    "time_to_first_token": "首 Token 延迟 (TTFT)",
    "time_to_thinking_end": "思考完成延迟",
    "total_inference_time": "总推理时间",
}

# English messages
MESSAGES_EN = {
    "thinking": "Thinking",
    "action": "Action",
    "task_completed": "Task Completed",
    "done": "Done",
    "starting_task": "Starting task",
    "final_result": "Final Result",
    "task_result": "Task Result",
    "confirmation_required": "Confirmation Required",
    "continue_prompt": "Continue? (y/n)",
    "manual_operation_required": "Manual Operation Required",
    "manual_operation_hint": "Please complete the operation manually...",
    "press_enter_when_done": "Press Enter when done",
    "connection_failed": "Connection Failed",
    "connection_successful": "Connection Successful",
    "step": "Step",
    "task": "Task",
    "result": "Result",
    "performance_metrics": "Performance Metrics",
    "time_to_first_token": "Time to First Token (TTFT)",
    "time_to_thinking_end": "Time to Thinking End",
    "total_inference_time": "Total Inference Time",
}


def get_messages(lang: str = "cn") -> dict:
    """
    Get UI messages dictionary by language.

    Args:
        lang: Language code, 'cn' for Chinese, 'en' for English.

    Returns:
        Dictionary of UI messages.
    """
    if lang == "en":
        return MESSAGES_EN
    return MESSAGES_ZH


def get_message(key: str, lang: str = "cn") -> str:
    """
    Get a single UI message by key and language.

    Args:
        key: Message key.
        lang: Language code, 'cn' for Chinese, 'en' for English.

    Returns:
        Message string.
    """
    messages = get_messages(lang)
    return messages.get(key, key)


today = datetime.today()
weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
weekday = weekday_names[today.weekday()]
formatted_date = today.strftime("%Y年%m月%d日") + " " + weekday

SYSTEM_PROMPT = (
    "今天的日期是: "
    + formatted_date
    + """
你是一个智能体分析专家，可以根据操作历史和当前状态图执行一系列操作来完成任务。
你必须严格按照要求输出以下格式：
<think>{think}</think>
<answer>{action}</answer>

其中：
- {think} 是对你为什么选择这个操作的简短推理说明。
- {action} 是本次执行的具体操作指令，必须严格遵循下方定义的指令格式。

操作指令及其作用如下：
- do(action="Launch", app="xxx")  
    Launch是启动目标app的操作，这比通过主屏幕导航更快。此操作完成后，您将自动收到结果状态的截图。
- do(action="Tap", element=[x,y])  
    Tap是点击操作，点击屏幕上的特定点。可用此操作点击按钮、选择项目、从主屏幕打开应用程序，或与任何可点击的用户界面元素进行交互。坐标系统从左上角 (0,0) 开始到右下角（999,999)结束。此操作完成后，您将自动收到结果状态的截图。
- do(action="Tap", element=[x,y], message="重要操作")  
    基本功能同Tap，点击涉及财产、支付、隐私等敏感按钮时触发。
- do(action="Type", text="xxx")  
    Type是输入操作，在当前聚焦的输入框中输入文本。使用此操作前，请确保输入框已被聚焦（先点击它）。输入的文本将像使用键盘输入一样输入。重要提示：手机可能正在使用 ADB 键盘，该键盘不会像普通键盘那样占用屏幕空间。要确认键盘已激活，请查看屏幕底部是否显示 'ADB Keyboard {ON}' 类似的文本，或者检查输入框是否处于激活/高亮状态。不要仅仅依赖视觉上的键盘显示。自动清除文本：当你使用输入操作时，输入框中现有的任何文本（包括占位符文本和实际输入）都会在输入新文本前自动清除。你无需在输入前手动清除文本——直接使用输入操作输入所需文本即可。操作完成后，你将自动收到结果状态的截图。
- do(action="Type_Name", text="xxx")  
    Type_Name是输入人名的操作，基本功能同Type。
- do(action="Interact")  
    Interact是当有多个满足条件的选项时而触发的交互操作，询问用户如何选择。
- do(action="Swipe", start=[x1,y1], end=[x2,y2])  
    Swipe是滑动操作，通过从起始坐标拖动到结束坐标来执行滑动手势。可用于滚动内容、在屏幕之间导航、下拉通知栏以及项目栏或进行基于手势的导航。坐标系统从左上角 (0,0) 开始到右下角（999,999)结束。滑动持续时间会自动调整以实现自然的移动。此操作完成后，您将自动收到结果状态的截图。
- do(action="Note", message="True")  
    记录当前页面内容以便后续总结。
- do(action="Call_API", instruction="xxx")  
    总结或评论当前页面或已记录的内容。
- do(action="Long Press", element=[x,y])  
    Long Pres是长按操作，在屏幕上的特定点长按指定时间。可用于触发上下文菜单、选择文本或激活长按交互。坐标系统从左上角 (0,0) 开始到右下角（999,999)结束。此操作完成后，您将自动收到结果状态的屏幕截图。
- do(action="Double Tap", element=[x,y])  
    Double Tap在屏幕上的特定点快速连续点按两次。使用此操作可以激活双击交互，如缩放、选择文本或打开项目。坐标系统从左上角 (0,0) 开始到右下角（999,999)结束。此操作完成后，您将自动收到结果状态的截图。
- do(action="Take_over", message="xxx")  
    Take_over是接管操作，表示在登录和验证阶段需要用户协助。
- do(action="Back")  
    导航返回到上一个屏幕或关闭当前对话框。相当于按下 Android 的返回按钮。使用此操作可以从更深的屏幕返回、关闭弹出窗口或退出当前上下文。此操作完成后，您将自动收到结果状态的截图。
- do(action="Home") 
    Home是回到系统桌面的操作，相当于按下 Android 主屏幕按钮。使用此操作可退出当前应用并返回启动器，或从已知状态启动新任务。此操作完成后，您将自动收到结果状态的截图。
- do(action="Wait", duration="x seconds")  
    等待页面加载，x为需要等待多少秒。
- finish(message="xxx")  
    finish是结束任务的操作，表示准确完整完成任务，message是终止信息。 

必须遵循的规则：
1. 在执行任何操作前，先检查当前app是否是目标app，如果不是，先执行 Launch。
2. 如果进入到了无关页面，先执行 Back。如果执行Back后页面没有变化，请点击页面左上角的返回键进行返回，或者右上角的X号关闭。
3. 如果页面未加载出内容，最多连续 Wait 三次，否则执行 Back重新进入。
4. 如果页面显示网络问题，需要重新加载，请点击重新加载。
5. 如果当前页面找不到目标联系人、商品、店铺等信息，可以尝试 Swipe 滑动查找。
6. 遇到价格区间、时间区间等筛选条件，如果没有完全符合的，可以放宽要求。
7. 在做小红书总结类任务时一定要筛选图文笔记。
8. 购物车全选后再点击全选可以把状态设为全不选，在做购物车任务时，如果购物车里已经有商品被选中时，你需要点击全选后再点击取消全选，再去找需要购买或者删除的商品。
9. 在做外卖任务时，如果相应店铺购物车里已经有其他商品你需要先把购物车清空再去购买用户指定的外卖。
10. 在做点外卖任务时，如果用户需要点多个外卖，请尽量在同一店铺进行购买，如果无法找到可以下单，并说明某个商品未找到。
11. 请严格遵循用户意图执行任务，用户的特殊要求可以执行多次搜索，滑动查找。比如（i）用户要求点一杯咖啡，要咸的，你可以直接搜索咸咖啡，或者搜索咖啡后滑动查找咸的咖啡，比如海盐咖啡。（ii）用户要找到XX群，发一条消息，你可以先搜索XX群，找不到结果后，将"群"字去掉，搜索XX重试。（iii）用户要找到宠物友好的餐厅，你可以搜索餐厅，找到筛选，找到设施，选择可带宠物，或者直接搜索可带宠物，必要时可以使用AI搜索。
12. 在选择日期时，如果原滑动方向与预期日期越来越远，请向反方向滑动查找。
13. 执行任务过程中如果有多个可选择的项目栏，请逐个查找每个项目栏，直到完成任务，一定不要在同一项目栏多次查找，从而陷入死循环。
14. 在执行下一步操作前请一定要检查上一步的操作是否生效，如果点击没生效，可能因为app反应较慢，请先稍微等待一下，如果还是不生效请调整一下点击位置重试，如果仍然不生效请跳过这一步继续任务，并在finish message说明点击不生效。
15. 在执行任务中如果遇到滑动不生效的情况，请调整一下起始点位置，增大滑动距离重试，如果还是不生效，有可能是已经滑到底了，请继续向反方向滑动，直到顶部或底部，如果仍然没有符合要求的结果，请跳过这一步继续任务，并在finish message说明但没找到要求的项目。
16. 在做游戏任务时如果在战斗页面如果有自动战斗一定要开启自动战斗，如果多轮历史状态相似要检查自动战斗是否开启。
17. 如果没有合适的搜索结果，可能是因为搜索页面不对，请返回到搜索页面的上一级尝试重新搜索，如果尝试三次返回上一级搜索后仍然没有符合要求的结果，执行 finish(message="原因")。
18. 在结束任务前请一定要仔细检查任务是否完整准确的完成，如果出现错选、漏选、多选的情况，请返回之前的步骤进行纠正。
"""
)

def encode_image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"{b64}"
class MessageBuilder:
    """Helper class for building conversation messages."""

    @staticmethod
    def create_system_message(content: str) -> dict[str, Any]:
        """Create a system message."""
        return {"role": "system", "content": content}

    @staticmethod
    def create_user_message(
        text: str, image_base64: str | None = None
    ) -> dict[str, Any]:
        """
        Create a user message with optional image.

        Args:
            text: Text content.
            image_base64: Optional base64-encoded image.

        Returns:
            Message dictionary.
        """
        content = []

        if image_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                }
            )

        content.append({"type": "text", "text": text})

        return {"role": "user", "content": content}

    @staticmethod
    def create_assistant_message(content: str) -> dict[str, Any]:
        """Create an assistant message."""
        return {"role": "assistant", "content": content}

    @staticmethod
    def remove_images_from_message(message: dict[str, Any]) -> dict[str, Any]:
        """
        Remove image content from a message to save context space.

        Args:
            message: Message dictionary.

        Returns:
            Message with images removed.
        """
        if isinstance(message.get("content"), list):
            message["content"] = [
                item for item in message["content"] if item.get("type") == "text"
            ]
        return message

    @staticmethod
    def build_screen_info(current_app: str, **extra_info) -> str:
        """
        Build screen info string for the model.

        Args:
            current_app: Current app name.
            **extra_info: Additional info to include.

        Returns:
            JSON string with screen info.
        """
        info = {"current_app": current_app, **extra_info}
        return json.dumps(info, ensure_ascii=False)


class ModelClient:
    """
    Client for interacting with OpenAI-compatible vision-language models.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    def request(self, messages: list[dict[str, Any]]) -> ModelResponse:
        """
        Send a request to the model.

        Args:
            messages: List of message dictionaries in OpenAI format.

        Returns:
            ModelResponse containing thinking and action.

        Raises:
            ValueError: If the response cannot be parsed.
        """
        # Start timing
        start_time = time.time()
        time_to_first_token = None
        time_to_thinking_end = None

        stream = self.client.chat.completions.create(
            messages=messages,
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            extra_body=self.config.extra_body,
            stream=True,
        )

        raw_content = ""
        buffer = ""  # Buffer to hold content that might be part of a marker
        action_markers = ["finish(message=", "do(action="]
        in_action_phase = False  # Track if we've entered the action phase
        first_token_received = False

        for chunk in stream:
            if len(chunk.choices) == 0:
                continue
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                raw_content += content

                # Record time to first token
                if not first_token_received:
                    time_to_first_token = time.time() - start_time
                    first_token_received = True

                if in_action_phase:
                    # Already in action phase, just accumulate content without printing
                    continue

                buffer += content

                # Check if any marker is fully present in buffer
                marker_found = False
                for marker in action_markers:
                    if marker in buffer:
                        # Marker found, print everything before it
                        thinking_part = buffer.split(marker, 1)[0]
                        print(thinking_part, end="", flush=True)
                        print()  # Print newline after thinking is complete
                        in_action_phase = True
                        marker_found = True

                        # Record time to thinking end
                        if time_to_thinking_end is None:
                            time_to_thinking_end = time.time() - start_time

                        break

                if marker_found:
                    continue  # Continue to collect remaining content

                # Check if buffer ends with a prefix of any marker
                # If so, don't print yet (wait for more content)
                is_potential_marker = False
                for marker in action_markers:
                    for i in range(1, len(marker)):
                        if buffer.endswith(marker[:i]):
                            is_potential_marker = True
                            break
                    if is_potential_marker:
                        break

                if not is_potential_marker:
                    # Safe to print the buffer
                    print(buffer, end="", flush=True)
                    buffer = ""

        # Calculate total time
        total_time = time.time() - start_time

        # Parse thinking and action from response
        thinking, action = self._parse_response(raw_content)

        # Print performance metrics
        lang = self.config.lang
        print()
        print("=" * 50)
        print(f"⏱️  {get_message('performance_metrics', lang)}:")
        print("-" * 50)
        if time_to_first_token is not None:
            print(
                f"{get_message('time_to_first_token', lang)}: {time_to_first_token:.3f}s"
            )
        if time_to_thinking_end is not None:
            print(
                f"{get_message('time_to_thinking_end', lang)}:        {time_to_thinking_end:.3f}s"
            )
        print(
            f"{get_message('total_inference_time', lang)}:          {total_time:.3f}s"
        )
        print("=" * 50)

        return ModelResponse(
            thinking=thinking,
            action=action,
            raw_content=raw_content,
            time_to_first_token=time_to_first_token,
            time_to_thinking_end=time_to_thinking_end,
            total_time=total_time,
        )

    def _parse_response(self, content: str) -> tuple[str, str]:
        """
        Parse the model response into thinking and action parts.

        Parsing rules:
        1. If content contains 'finish(message=', everything before is thinking,
           everything from 'finish(message=' onwards is action.
        2. If rule 1 doesn't apply but content contains 'do(action=',
           everything before is thinking, everything from 'do(action=' onwards is action.
        3. Fallback: If content contains '<answer>', use legacy parsing with XML tags.
        4. Otherwise, return empty thinking and full content as action.

        Args:
            content: Raw response content.

        Returns:
            Tuple of (thinking, action).
        """
        # Rule 1: Check for finish(message=
        if "finish(message=" in content:
            parts = content.split("finish(message=", 1)
            thinking = parts[0].strip()
            action = "finish(message=" + parts[1]
            return thinking, action

        # Rule 2: Check for do(action=
        if "do(action=" in content:
            parts = content.split("do(action=", 1)
            thinking = parts[0].strip()
            action = "do(action=" + parts[1]
            return thinking, action

        # Rule 3: Fallback to legacy XML tag parsing
        if "<answer>" in content:
            parts = content.split("<answer>", 1)
            thinking = parts[0].replace("<think>", "").replace("</think>", "").strip()
            action = parts[1].replace("</answer>", "").strip()
            return thinking, action

        # Rule 4: No markers found, return content as action
        return "", content

def parse_action(response: str) -> dict[str, Any]:
    """
    Parse action from model response.

    Args:
        response: Raw response string from the model.

    Returns:
        Parsed action dictionary.

    Raises:
        ValueError: If the response cannot be parsed.
    """
    try:
        response = response.strip()
        if response.startswith('do(action="Type"') or response.startswith(
            'do(action="Type_Name"'
        ):
            text = response.split("text=", 1)[1][1:-2]
            action = {"_metadata": "do", "action": "Type", "text": text}
            return action
        elif response.startswith("do"):
            # Use AST parsing instead of eval for safety
            try:
                tree = ast.parse(response, mode="eval")
                if not isinstance(tree.body, ast.Call):
                    raise ValueError("Expected a function call")

                call = tree.body
                # Extract keyword arguments safely
                action = {"_metadata": "do"}
                for keyword in call.keywords:
                    key = keyword.arg
                    value = ast.literal_eval(keyword.value)
                    action[key] = value

                return action
            except (SyntaxError, ValueError) as e:
                raise ValueError(f"Failed to parse do() action: {e}")

        elif response.startswith("finish"):
            action = {
                "_metadata": "finish",
                "message": response.replace("finish(message=", "")[1:-2],
            }
        else:
            raise ValueError(f"Failed to parse action: {response}")
        return action
    except Exception as e:
        raise ValueError(f"Failed to parse action: {e}")

def get_img_w_h(img_path):
    img = Image.open(img_path)
    return img.width,img.height

def get_std_act(action,width,height):
    action_type = action.get("_metadata")
    action_name = action.get("action")

    if action_type != "do" and action_type != "finish":
        return None
    else:
        if action_name in ["Tap"]:
            element = action.get("element")
            x = int(element[0] / 1000 * width)
            y = int(element[1] / 1000 * height)

            return Action(act_type="click", parameters={"position_x": x, "position_y": y})
        elif action_name in ["Type"]:
            text = action.get("text", "")

            return Action(act_type="input", parameters={"text": text})
        
        elif action_name in ["Swipe"]:
            start = action.get("start")
            end = action.get("end")
            x1, y1 = int(start[0] / 1000 * width),int(start[1] / 1000 * height)
            x2, y2 = int(end[0] / 1000 * width),int(end[1] / 1000 * height)
            if abs(x2 - x1) > abs(y2 - y1):
                direction = "right" if x2 > x1 else "left"
            else:
                direction = "down" if y2 > y1 else "up"
            dir_ = direction.lower()
            std_action = Action(act_type="swipe", parameters={"direction":dir_,"start_x": x1, "start_y": y1, "end_x": x2, "end_y": y2})

            return std_action
        
        elif action_name in ["Back"]:
            return Action(act_type="back", parameters={})
        elif action_name in ["Home"]:
            return Action(act_type="home", parameters={})
        elif action_name in ["Wait"]:
            return Action(act_type="wait", parameters={})
        else:
            return None

    

def run(
    fsm,
    args,
    app: str,
    task: str,
    instruction: str,
    runs_dir: str,
    model,
    model_config,
    verbose = True
):
    max_steps = fsm.max_op_times
    _context = []
    model_client = ModelClient(model_config)

    for step in range(1,max_steps,1):
        cur = fsm.cur_state
        cur_img_b64 = encode_image_to_b64(cur.img_path)
        img_w,img_h = get_img_w_h(cur.img_path)
        print(f"w {img_w} {img_h}")
        if step == 1:
            _context.append(
                MessageBuilder.create_system_message(SYSTEM_PROMPT)
            )

            screen_info = MessageBuilder.build_screen_info(app)
            text_content = f"{instruction}\n\n{screen_info}"

            _context.append(
                MessageBuilder.create_user_message(
                    text=text_content, image_base64=cur_img_b64
                )
            )
        else:
            screen_info = MessageBuilder.build_screen_info(app)
            text_content = f"** Screen Info **\n\n{screen_info}"

            _context.append(
                MessageBuilder.create_user_message(
                    text=text_content, image_base64=cur_img_b64
                )
            )
        

        try:
            msgs = get_messages()
            print("\n" + "=" * 50)
            print(f"💭 {msgs['thinking']}:")
            print("-" * 50)
            response = model_client.request(_context)
        except Exception as e:
            if verbose:
                traceback.print_exc()
            print("[1] ERR Finished ! ")
            return 

        # Parse action from response
        try:
            action = parse_action(response.action)
        except ValueError:
            if verbose:
                traceback.print_exc()
            print("[2] ERR Finished ! ")
            return
        if verbose:
            # Print thinking process
            print("-" * 50)
            print(f"🎯 {msgs['action']}:")
            print(json.dumps(action, ensure_ascii=False, indent=2))
            print("=" * 50 + "\n")

        # Remove image from context to save space
        _context[-1] = MessageBuilder.remove_images_from_message(_context[-1])
        #print(action)
        # Execute action
        try:
            std_act = get_std_act(
                action, img_w, img_h
            )
        except Exception as e:
            if verbose:
                traceback.print_exc()
            # result = self.action_handler.execute(
            #     finish(message=str(e)), screenshot.width, screenshot.height
            # )

        # Add assistant response to context
        _context.append(
            MessageBuilder.create_assistant_message(
                f"<think>{response.thinking}</think><answer>{response.action}</answer>"
            )
        )

        # Check if finished
        finished = action.get("_metadata") == "finish" 

        if finished and verbose:
            msgs = get_messages()
            print("\n" + "🎉 " + "=" * 48)
            print(
                f"✅ {msgs['task_completed']}"
            )
            print("=" * 50 + "\n")
            break

        elif std_act == None:
            print("task failed !")
            break
        
        prev_state = fsm.cur_state
        fsm.action(std_act)
        new_state = fsm.cur_state
        if new_state.cluster_class in ("DONE", "Done", "done"):
            print("到达 DONE 状态，任务完成！")
           
            break

        if fsm.is_failed:
            print("到达 FAILED 状态，任务失败！")
            break
    print("finished state",fsm.cur_state.img_path)





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Auto collection of GUI data')
    parser.add_argument('--model', type=str,  help='name of the LLM model')
    parser.add_argument('--api_key', type=str,  help='API key for the LLM model')
    parser.add_argument('--base_url', type=str, help='base URL for the LLM model API')
    parser.add_argument('--max_steps', type=int, default=15, help='maximum steps per task (default: 15)')
    parser.add_argument("--data_root", default="/Users/fengyunfei/Desktop/mobiagent/MobiBench/data", help="MobiBench data 根目录（包含 rawdata/）")
    parser.add_argument(
        "--runs_dir",
        default=r"/Users/fengyunfei/Desktop/mobiagent/MobiBench/agents/autoglm/layers",  # ==== NEW: 所有运行结果的根目录 ====
        help="所有运行结果的根目录，用于保存轨迹和坐标",
    )
    args = parser.parse_args()
    
    # 设置全局配置
    model = args.model
    api_key = args.api_key
    base_url = args.base_url
    max_steps = args.max_steps
    md_cfg = ModelConfig()
    # 初始化OpenAI客户端
    # client = OpenAI(
    #         api_key= "sk-rfCIGhxrzcdsMV4jC17e406bE56c47CbA5416068A62318D3",
    #         base_url=f"http://ipads.chat.gpt:3006/v1"
    #     )
    with open('/Users/fengyunfei/Desktop/mobiagent/MobiBench/data/base.json', 'r', encoding='utf-8') as f:
        alldata = json.load(f)
    datapath = args.data_root
    data_log_dir = "/Users/fengyunfei/Desktop/mobiagent/MobiBench/agents/autoglm/log"
    for app in alldata.keys():
        for tasktype in alldata[app]:
            tasklist = get_tasks(app, tasktype)
            #logger.info("构建 FSM 中…")
            fsm = build_AppFSM(app=app, task=tasktype, data_path=datapath)
            # 让 FSM 内部的 max_op_times 和 CLI 一致
            fsm.max_op_times = args.max_steps

            for task in tasklist:
                print(f"任务: {task}，应用: {app}，类型: {tasktype}")
                #logger = setup_logger(data_log_dir)
                #logger.info("程序启动")
                
                fsm._reset()
                
                start = time.time()
                run(
                    fsm=fsm,
                    args=args,
                    app=app,
                    task=tasktype,
                    instruction=task,
                    runs_dir=args.runs_dir,  # ==== NEW: 传入 runs 根目录 ====
                    
                    model="autoglm",
                    model_config=md_cfg

                )
                end = time.time()
                from MobiBench.utils.score_proc import save_result
                save_result(
                    md="autoglm",
                    app=app,
                    task=tasktype,
                    inst=task,
                    fsm=fsm,
                    time_use=end-start,
                    savepath=r"/Users/fengyunfei/Desktop/mobiagent/MobiBench/results/dev",
                )
                from MobiBench.utils.score_proc import save_visited_result
                save_visited_result(
                    md="autoglm",
                    app=app,
                    task=tasktype,
                    fsm=fsm,
                    savepath=r"/Users/fengyunfei/Desktop/mobiagent/MobiBench/results/dev/visited",
                )
    
