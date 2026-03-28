#!/usr/bin/env python3
"""
中文自动评估脚本：
1) 扫描日志目录 (fsm_eval_trace.json)。
2) 构建运行摘要与聚合指标。
3) 提供丰富的 function-calling 工具（中文描述），LLM 可按需查看失败步骤、截图路径、任务上下文等。
4) LLM 以中文输出对比结论、特点与改进建议。

默认目录：
- UI_TARS: /home/fff/mobibench/MobiBench/agents/UI_TARS/runs/20260316
- MobiMind: /home/fff/mobibench/MobiBench/agents/MobiMind/data/20260315
"""

import argparse
import base64
import io
import json
import logging
import os
import re
import statistics
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List


# ----------------------------- data structures -----------------------------

@dataclass
class RunSummary:
    run_id: str
    model: str
    app: str
    task: str
    success: bool
    steps: int
    clicks: int
    swipes: int
    inputs: int
    first_img: str
    last_img: str
    path: str


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def setup_logger(log_file: str = "") -> logging.Logger:
    logger = logging.getLogger("log_eval_auto")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def save_markdown_report(
    output_path: str,
    aggregates: Dict[str, Dict[str, float]],
    dimensional_aggregates: Dict[str, Dict[str, dict]],
    final_text: str,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    content = [
        "# GUI 智能体自动评估报告",
        "",
        f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 聚合指标",
        "```json",
        json.dumps(aggregates, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 分维度指标",
        "```json",
        json.dumps(dimensional_aggregates, ensure_ascii=False, indent=2),
        "```",
        "",
        "## LLM 结论",
        "",
        final_text or "(空输出)",
        "",
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content))


def _speak_with_pyttsx3(text: str, out_wav: str):
    import pyttsx3

    engine = pyttsx3.init()
    engine.save_to_file(text, out_wav)
    engine.runAndWait()


def _shell_ok(cmd: List[str]) -> bool:
    return subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def save_tts_audio(text: str, output_path: str, logger: logging.Logger) -> bool:
    if not text.strip():
        logger.warning("[tts] 空文本，跳过语音生成")
        return False
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 方案 1: pyttsx3（纯本地）
    try:
        _speak_with_pyttsx3(text, output_path)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"[tts] 语音已生成（pyttsx3）: {output_path}")
            return True
    except Exception as e:
        logger.warning(f"[tts] pyttsx3 失败: {e}")

    # 方案 2: espeak
    try:
        if _shell_ok(["which", "espeak"]):
            cmd = ["espeak", "-v", "zh", "-w", output_path, text[:800]]
            subprocess.run(cmd, check=True)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"[tts] 语音已生成（espeak）: {output_path}")
                return True
    except Exception as e:
        logger.warning(f"[tts] espeak 失败: {e}")

    # 方案 3: ffmpeg flite（部分环境可用）
    try:
        if _shell_ok(["which", "ffmpeg"]):
            safe_text = text[:500].replace("'", " ")
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"flite=text='{safe_text}':voice=slt",
                output_path,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"[tts] 语音已生成（ffmpeg-flite）: {output_path}")
                return True
    except Exception as e:
        logger.warning(f"[tts] ffmpeg-flite 失败: {e}")

    logger.warning("[tts] 所有语音方案均失败，已跳过")
    return False


# ----------------------------- log parsing -----------------------------

def _safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def infer_app_type(app: str) -> str:
    table = {
        "微博": "社交媒体",
        "小红书": "社交媒体",
        "知乎": "内容社区",
        "网易云音乐": "音乐娱乐",
        "高德": "地图出行",
        "同城": "出行服务",
        "美团": "本地生活",
        "淘宝": "电商购物",
        "饿了么": "本地生活",
        "携程": "出行服务",
        "飞书": "办公协作",
        "腾讯会议": "办公协作",
        "多邻国": "教育学习",
    }
    return table.get(app, "其他应用")


def infer_task_intent(task: str, instruction: str) -> str:
    text = f"{task} {instruction or ''}"
    rules = [
        ("隐私设置", r"隐私|权限|设置|开屏|广告|通知"),
        ("搜索与浏览", r"搜索|查找|打开.*笔记|主页|浏览"),
        ("输入与发布", r"输入|评论|发送|发布|填写"),
        ("下单与交易", r"下单|支付|购买|购物车|订单"),
        ("导航与出行", r"导航|路线|打车|机票|火车|酒店"),
        ("媒体播放", r"播放|歌曲|音乐|视频"),
    ]
    for intent, pat in rules:
        if re.search(pat, text):
            return intent
    return "通用操作"


def _img_diff_score(path_a: str, path_b: str, size: int = 48) -> float:
    """
    计算两张截图的粗粒度差异分值，范围约 [0, 1]；分值越小越接近。
    """
    try:
        from PIL import Image

        with Image.open(path_a) as ia, Image.open(path_b) as ib:
            a = ia.convert("L").resize((size, size))
            b = ib.convert("L").resize((size, size))
            pa = list(a.getdata())
            pb = list(b.getdata())
        if not pa or len(pa) != len(pb):
            return 1.0
        diff = sum(abs(x - y) for x, y in zip(pa, pb)) / (255.0 * len(pa))
        return float(diff)
    except Exception:
        return 1.0


def _is_done_label(label: str) -> bool:
    return str(label or "").strip().lower() == "done"


def _extract_thought(step: dict) -> str:
    t = step.get("thought")
    if isinstance(t, str) and t.strip():
        return t.strip()
    raw = step.get("raw_output")
    if isinstance(raw, dict):
        txt = raw.get("reasoning")
        return txt.strip() if isinstance(txt, str) else ""
    if isinstance(raw, str):
        m = re.search(r"Thought:\s*(.*?)(?:\nAction:|$)", raw, flags=re.S)
        if m:
            return m.group(1).strip()
    return ""


def load_run(trace_path: str, model: str) -> RunSummary:
    with open(trace_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    steps = data.get("steps", [])
    clicks = sum(1 for s in steps if s.get("action_type") == "click")
    swipes = sum(1 for s in steps if s.get("action_type") in ("swipe", "drag"))
    inputs = sum(1 for s in steps if s.get("action_type") in ("input", "type"))
    first_img = steps[0]["prev_img"] if steps else ""
    last_img = steps[-1]["new_img"] if steps else ""
    return RunSummary(
        run_id=f"{model}:{data.get('app','?')}/{data.get('task','?')}/{os.path.basename(os.path.dirname(trace_path))}",
        model=model,
        app=data.get("app", "?"),
        task=data.get("task", "?"),
        success=bool(data.get("success")),
        steps=_safe_int(len(steps)),
        clicks=clicks,
        swipes=swipes,
        inputs=inputs,
        first_img=first_img,
        last_img=last_img,
        path=trace_path,
    )


def scan_runs(root: str, model: str) -> Dict[str, RunSummary]:
    idx: Dict[str, RunSummary] = {}
    for dirpath, _, files in os.walk(root):
        if "fsm_eval_trace.json" in files:
            run_path = os.path.join(dirpath, "fsm_eval_trace.json")
            try:
                summary = load_run(run_path, model)
                idx[summary.run_id] = summary
            except Exception as e:
                print(f"[warn] skip {run_path}: {e}")
    return idx


def aggregate(summaries: List[RunSummary]) -> Dict[str, float]:
    if not summaries:
        return {}
    success_rate = sum(1 for s in summaries if s.success) / len(summaries)
    avg_steps = statistics.mean(s.steps for s in summaries) if summaries else 0.0
    return {
        "runs": len(summaries),
        "success_rate": round(success_rate, 3),
        "avg_steps": round(avg_steps, 2),
        "avg_clicks": round(statistics.mean(s.clicks for s in summaries), 2),
        "avg_swipes": round(statistics.mean(s.swipes for s in summaries), 2),
        "avg_inputs": round(statistics.mean(s.inputs for s in summaries), 2),
    }


def aggregate_with_dimensions(summaries: List[RunSummary], traces: Dict[str, dict]) -> Dict[str, dict]:
    def bucket_stats(items: List[RunSummary]) -> Dict[str, float]:
        return aggregate(items)

    out = {"global": bucket_stats(summaries), "by_app": {}, "by_app_type": {}, "by_task": {}, "by_intent": {}}
    by_app: Dict[str, List[RunSummary]] = defaultdict(list)
    by_app_type: Dict[str, List[RunSummary]] = defaultdict(list)
    by_task: Dict[str, List[RunSummary]] = defaultdict(list)
    by_intent: Dict[str, List[RunSummary]] = defaultdict(list)

    for s in summaries:
        by_app[s.app].append(s)
        by_app_type[infer_app_type(s.app)].append(s)
        by_task[s.task].append(s)
        tr = traces.get(s.run_id, {})
        by_intent[infer_task_intent(s.task, tr.get("instruction", ""))].append(s)

    out["by_app"] = {k: bucket_stats(v) for k, v in by_app.items()}
    out["by_app_type"] = {k: bucket_stats(v) for k, v in by_app_type.items()}
    out["by_task"] = {k: bucket_stats(v) for k, v in by_task.items()}
    out["by_intent"] = {k: bucket_stats(v) for k, v in by_intent.items()}
    return out


def diagnose_trace(trace: dict) -> dict:
    steps = trace.get("steps", []) or []
    if not steps:
        return {
            "app_type": infer_app_type(trace.get("app", "")),
            "task_intent": infer_task_intent(trace.get("task", ""), trace.get("instruction", "")),
            "first_error_step": None,
            "possible_causes": ["空轨迹或解析失败"],
            "metrics": {},
            "key_errors": [],
        }

    stagnation = 0
    no_change_click = 0
    no_change_swipe = 0
    no_change_input = 0
    premature_done = 0
    repeated_action_max = 1
    current_streak = 1
    first_error_step = None
    key_errors = []
    repeated_thought = 0
    weak_thought = 0
    prev_thought = ""

    prev_sig = None
    for idx, s in enumerate(steps):
        step_no = s.get("step", idx + 1)
        act = (s.get("action_type") or "").lower()
        thought = _extract_thought(s)
        if thought and thought == prev_thought:
            repeated_thought += 1
        prev_thought = thought or prev_thought
        if (not thought) or len(thought) < 8:
            weak_thought += 1
        params = s.get("action_params") or {}
        sig = f"{act}|{json.dumps(params, sort_keys=True, ensure_ascii=False)}"
        if prev_sig == sig:
            current_streak += 1
        else:
            current_streak = 1
        repeated_action_max = max(repeated_action_max, current_streak)
        prev_sig = sig

        same_label = s.get("prev_label") == s.get("new_label")
        same_path = s.get("prev_img") == s.get("new_img")
        diff = _img_diff_score(s.get("prev_img", ""), s.get("new_img", ""))
        no_visual_change = diff < 0.01
        no_progress = same_label or same_path or no_visual_change

        if no_progress:
            stagnation += 1
            if first_error_step is None:
                first_error_step = step_no
            if act == "click":
                no_change_click += 1
            elif act == "swipe":
                no_change_swipe += 1
            elif act in ("input", "click_input", "type"):
                no_change_input += 1

        if act == "done" and (not _is_done_label(s.get("new_label"))) and (not trace.get("success", False)):
            premature_done += 1
            key_errors.append(
                {
                    "step": step_no,
                    "reason": "提前结束：动作为 done，但状态未到 DONE 且任务失败",
                    "prev_label": s.get("prev_label"),
                    "new_label": s.get("new_label"),
                    "img": s.get("new_img"),
                }
            )

        if no_progress and current_streak >= 3:
            key_errors.append(
                {
                    "step": step_no,
                    "reason": "重复动作且页面无变化，疑似卡住/点击无效",
                    "prev_label": s.get("prev_label"),
                    "new_label": s.get("new_label"),
                    "img": s.get("new_img"),
                }
            )

    causes = []
    if stagnation >= max(2, len(steps) // 2):
        causes.append("高比例无进展步骤（页面/状态变化很小），可能是目标定位不准或交互策略单一")
    if no_change_click >= 3:
        causes.append("点击动作多次无效，可能点击坐标偏移、控件不可点击或需先滚动后点击")
    if no_change_swipe >= 3:
        causes.append("滑动动作多次无效，可能方向错误或页面已到边界")
    if no_change_input >= 2:
        causes.append("输入相关动作未触发状态变化，可能未正确聚焦输入框或缺少确认提交动作")
    if repeated_action_max >= 4:
        causes.append("存在长重复动作序列，疑似陷入局部循环")
    if premature_done > 0:
        causes.append("模型出现过早 done，终止条件判断可能过于乐观")
    if repeated_thought >= 3:
        causes.append("思考文本高度重复，可能缺少基于新观测的重规划能力")
    if weak_thought >= max(2, len(steps) // 3):
        causes.append("部分步骤思考信息量不足，可能影响定位与决策稳定性")
    if not causes:
        causes.append("主要由任务路径复杂度或个别步骤决策误差导致")

    return {
        "app_type": infer_app_type(trace.get("app", "")),
        "task_intent": infer_task_intent(trace.get("task", ""), trace.get("instruction", "")),
        "first_error_step": first_error_step,
        "possible_causes": causes,
        "metrics": {
            "total_steps": len(steps),
            "stagnation_steps": stagnation,
            "no_change_click": no_change_click,
            "no_change_swipe": no_change_swipe,
            "no_change_input": no_change_input,
            "premature_done": premature_done,
            "repeat_action_max_streak": repeated_action_max,
            "repeated_thought_steps": repeated_thought,
            "weak_thought_steps": weak_thought,
        },
        "key_errors": key_errors[:6],
    }


# ----------------------------- LLM tools & helpers -----------------------------

def load_trace(run_id: str, run_index: Dict[str, RunSummary]) -> dict:
    r = run_index.get(run_id)
    if not r:
        return {}
    try:
        with open(r.path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def build_tools(run_index: Dict[str, RunSummary], llm_client: Any = None, judge_model: str = ""):
    trace_cache: Dict[str, dict] = {}
    diag_cache: Dict[str, dict] = {}

    def _load_trace_cached(run_id: str) -> dict:
        if run_id in trace_cache:
            return trace_cache[run_id]
        trace = load_trace(run_id, run_index)
        trace_cache[run_id] = trace
        return trace

    def _diag_cached(run_id: str) -> dict:
        if run_id in diag_cache:
            return diag_cache[run_id]
        trace = _load_trace_cached(run_id)
        diag = diagnose_trace(trace)
        diag_cache[run_id] = diag
        return diag

    def list_runs():
        return [
            {"run_id": rid, "模型": r.model, "应用": r.app, "任务": r.task, "成功": r.success}
            for rid, r in run_index.items()
        ]

    def get_run(run_id: str):
        r = run_index.get(run_id)
        return asdict(r) if r else None

    def sample_failures(model: str, limit: int = 3):
        fails = [r for r in run_index.values() if r.model == model and not r.success]
        fails = sorted(fails, key=lambda x: x.steps, reverse=True)[:limit]
        return [asdict(r) for r in fails]

    def sample_successes(model: str, limit: int = 3):
        succ = [r for r in run_index.values() if r.model == model and r.success]
        succ = sorted(succ, key=lambda x: x.steps)[:limit]
        return [asdict(r) for r in succ]

    def get_steps(run_id: str, start: int = 0, limit: int = 10):
        trace = _load_trace_cached(run_id)
        steps = trace.get("steps", [])
        slice_steps = steps[start : start + limit]
        return slice_steps

    def get_task_context(run_id: str):
        trace = _load_trace_cached(run_id)
        return {
            "app": trace.get("app"),
            "task": trace.get("task"),
            "app_type": infer_app_type(trace.get("app", "")),
            "task_intent": infer_task_intent(trace.get("task", ""), trace.get("instruction", "")),
            "instruction": trace.get("instruction"),
            "success": trace.get("success"),
            "total_steps": len(trace.get("steps", [])),
        }

    def get_step_image(run_id: str, step: int, which: str = "prev", embed: bool = False, max_size: int = 512):
        trace = _load_trace_cached(run_id)
        steps = trace.get("steps", [])
        if step < 0 or step >= len(steps):
            return None
        img_path = steps[step].get("prev_img" if which == "prev" else "new_img", "")
        if not embed:
            return {"path": img_path}
        try:
            from PIL import Image

            with Image.open(img_path) as img:
                img.thumbnail((max_size, max_size))
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return {"path": img_path, "base64_jpeg": b64}
        except Exception as e:
            return {"path": img_path, "error": str(e)}

    def stats_by_task(model: str):
        data = {}
        for r in run_index.values():
            if r.model != model:
                continue
            key = r.task
            bucket = data.setdefault(key, {"runs": 0, "success": 0, "steps": []})
            bucket["runs"] += 1
            bucket["success"] += int(r.success)
            bucket["steps"].append(r.steps)
        for k, v in data.items():
            v["success_rate"] = round(v["success"] / v["runs"], 3) if v["runs"] else 0.0
            v["avg_steps"] = round(statistics.mean(v["steps"]), 2) if v["steps"] else 0.0
            v.pop("steps", None)
        return data

    def op_distribution(model: str):
        clicks = swipes = inputs = total = 0
        for r in run_index.values():
            if r.model != model:
                continue
            clicks += r.clicks
            swipes += r.swipes
            inputs += r.inputs
            total += r.steps
        return {
            "click_pct": round(clicks / total, 3) if total else 0.0,
            "swipe_pct": round(swipes / total, 3) if total else 0.0,
            "input_pct": round(inputs / total, 3) if total else 0.0,
            "total_steps": total,
        }

    def last_fail_reason(model: str, limit: int = 5):
        fails = [r for r in run_index.values() if r.model == model and not r.success]
        fails = fails[:limit]
        out = []
        for r in fails:
            trace = _load_trace_cached(r.run_id)
            steps = trace.get("steps", [])
            if not steps:
                continue
            tail = steps[-1]
            out.append(
                {
                    "run_id": r.run_id,
                    "app": r.app,
                    "task": r.task,
                    "last_action": tail.get("action_type"),
                    "last_params": tail.get("action_params"),
                    "last_label": tail.get("new_label"),
                    "last_img": tail.get("new_img"),
                }
            )
        return out

    def failure_slice(run_id: str, tail: int = 5):
        trace = _load_trace_cached(run_id)
        steps = trace.get("steps", [])
        if not steps:
            return []
        return steps[-tail:]

    def loop_signals(run_id: str):
        trace = _load_trace_cached(run_id)
        steps = trace.get("steps", [])
        labels = [s.get("new_label") for s in steps if s.get("new_label") is not None]
        imgs = [s.get("new_img") for s in steps if s.get("new_img") is not None]
        uniq_labels = len(set(labels)) if labels else 0
        uniq_imgs = len(set(imgs)) if imgs else 0
        is_loop = (len(labels) - uniq_labels) >= 2 or (len(imgs) - uniq_imgs) >= 2
        return {
            "steps": len(steps),
            "unique_labels": uniq_labels,
            "unique_imgs": uniq_imgs,
            "possible_loop": is_loop,
        }

    def error_position_hist(model: str):
        buckets = {}
        for r in run_index.values():
            if r.model != model or r.success:
                continue
            trace = _load_trace_cached(r.run_id)
            steps = trace.get("steps", [])
            if not steps:
                continue
            last = steps[-1]
            label = last.get("new_label") or "UNKNOWN"
            buckets[label] = buckets.get(label, 0) + 1
        return buckets

    def get_run_diagnosis(run_id: str):
        trace = _load_trace_cached(run_id)
        if not trace:
            return None
        return {
            "run_id": run_id,
            "summary": asdict(run_index[run_id]) if run_id in run_index else None,
            "diagnosis": _diag_cached(run_id),
        }

    def inspect_error_step(run_id: str, step: int):
        trace = _load_trace_cached(run_id)
        steps = trace.get("steps", [])
        if step < 0 or step >= len(steps):
            return None
        s = steps[step]
        diff = _img_diff_score(s.get("prev_img", ""), s.get("new_img", ""))
        no_change = (
            s.get("prev_label") == s.get("new_label")
            or s.get("prev_img") == s.get("new_img")
            or diff < 0.01
        )
        reason = "正常推进"
        if no_change and (s.get("action_type") or "").lower() == "click":
            reason = "点击后无变化，疑似未命中可交互元素"
        elif no_change and (s.get("action_type") or "").lower() == "swipe":
            reason = "滑动后无变化，疑似方向不对或已到边界"
        elif no_change:
            reason = "动作后无变化，可能是前置条件不满足"
        return {
            "step": s.get("step", step + 1),
            "action_type": s.get("action_type"),
            "action_params": s.get("action_params"),
            "prev_label": s.get("prev_label"),
            "new_label": s.get("new_label"),
            "prev_img": s.get("prev_img"),
            "new_img": s.get("new_img"),
            "pixel_diff": round(diff, 5),
            "no_progress": no_change,
            "candidate_reason": reason,
        }

    def compare_models_on_app(app: str):
        out = {}
        for m in ("UI_TARS", "MobiMind"):
            cur = [r for r in run_index.values() if r.model == m and r.app == app]
            out[m] = aggregate(cur)
        return {"app": app, "app_type": infer_app_type(app), "comparison": out}

    def compare_models_on_task(task: str):
        out = {}
        for m in ("UI_TARS", "MobiMind"):
            cur = [r for r in run_index.values() if r.model == m and r.task == task]
            out[m] = aggregate(cur)
        return {"task": task, "comparison": out}

    def model_error_taxonomy(model: str):
        rows = [r for r in run_index.values() if r.model == model and not r.success]
        tax = defaultdict(int)
        for r in rows:
            diag = _diag_cached(r.run_id)
            causes = diag.get("possible_causes", [])
            for c in causes:
                tax[c] += 1
        return dict(sorted(tax.items(), key=lambda x: x[1], reverse=True))

    def hard_cases(limit: int = 10):
        rows = []
        for r in run_index.values():
            if r.success:
                continue
            diag = _diag_cached(r.run_id)
            hard_score = diag.get("metrics", {}).get("stagnation_steps", 0) + diag.get("metrics", {}).get("repeat_action_max_streak", 0)
            rows.append(
                {
                    "run_id": r.run_id,
                    "model": r.model,
                    "app": r.app,
                    "task": r.task,
                    "app_type": infer_app_type(r.app),
                    "task_intent": infer_task_intent(r.task, _load_trace_cached(r.run_id).get("instruction", "")),
                    "hard_score": hard_score,
                    "first_error_step": diag.get("first_error_step"),
                    "possible_causes": diag.get("possible_causes", []),
                }
            )
        return sorted(rows, key=lambda x: x["hard_score"], reverse=True)[:limit]

    def llm_run_judge(run_id: str, with_images: bool = False):
        if llm_client is None:
            return {"error": "未配置 llm_client，无法执行 LLM 复核"}
        trace = _load_trace_cached(run_id)
        if not trace:
            return {"error": f"run_id 不存在: {run_id}"}
        diag = _diag_cached(run_id)
        steps = trace.get("steps", [])
        # 仅传关键片段，避免 token 过大
        short_steps = []
        for s in steps[:2] + steps[-4:]:
            short_steps.append(
                {
                    "step": s.get("step"),
                    "action_type": s.get("action_type"),
                    "prev_label": s.get("prev_label"),
                    "new_label": s.get("new_label"),
                    "thought": _extract_thought(s),
                    "prev_img": s.get("prev_img"),
                    "new_img": s.get("new_img"),
                }
            )
        if with_images:
            for s in short_steps:
                for key in ("prev_img", "new_img"):
                    p = s.get(key, "")
                    s[f"{key}_exists"] = bool(p and os.path.exists(p))

        prompt = {
            "instruction": trace.get("instruction"),
            "app": trace.get("app"),
            "task": trace.get("task"),
            "success": trace.get("success"),
            "rule_diagnosis": diag,
            "steps_excerpt": short_steps,
            "要求": "请输出 JSON，字段: error_stage, root_causes[], reasoning_issues[], action_issues[], termination_issues[], confidence(0-1), evidence_steps[]",
        }
        req = {
            "model": judge_model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "你是 GUI 智能体失败诊断专家。只输出 JSON，不要解释。",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        }
        try:
            resp = llm_client.chat.completions.create(
                response_format={"type": "json_object"},
                **req,
            )
        except Exception:
            resp = llm_client.chat.completions.create(**req)
        txt = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(txt)
        except Exception:
            data = {"raw": txt}
        return {"run_id": run_id, "llm_judgement": data}

    def llm_model_meta_review(model: str, limit: int = 6):
        if llm_client is None:
            return {"error": "未配置 llm_client，无法执行 LLM 复核"}
        fails = [r for r in run_index.values() if r.model == model and not r.success]
        fails = sorted(fails, key=lambda x: x.steps, reverse=True)[:limit]
        pack = []
        for r in fails:
            d = _diag_cached(r.run_id)
            pack.append(
                {
                    "run_id": r.run_id,
                    "app": r.app,
                    "task": r.task,
                    "app_type": infer_app_type(r.app),
                    "task_intent": infer_task_intent(r.task, _load_trace_cached(r.run_id).get("instruction", "")),
                    "first_error_step": d.get("first_error_step"),
                    "possible_causes": d.get("possible_causes", []),
                    "metrics": d.get("metrics", {}),
                }
            )

        req = {
            "model": judge_model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "你是 GUI 智能体评测专家。只输出 JSON，字段包含: model_traits[], common_failure_modes[], thinking_problems[], action_policy_problems[], prompt_fix_suggestions[], runtime_guardrail_suggestions[]",
                },
                {
                    "role": "user",
                    "content": json.dumps({"model": model, "cases": pack}, ensure_ascii=False),
                },
            ],
        }
        try:
            resp = llm_client.chat.completions.create(
                response_format={"type": "json_object"},
                **req,
            )
        except Exception:
            resp = llm_client.chat.completions.create(**req)
        txt = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(txt)
        except Exception:
            data = {"raw": txt}
        return {"model": model, "meta_review": data}

    return {
        "list_runs": list_runs,
        "get_run": get_run,
        "sample_failures": sample_failures,
        "sample_successes": sample_successes,
        "get_steps": get_steps,
        "get_task_context": get_task_context,
        "get_step_image": get_step_image,
        "stats_by_task": stats_by_task,
        "op_distribution": op_distribution,
        "last_fail_reason": last_fail_reason,
        "failure_slice": failure_slice,
        "loop_signals": loop_signals,
        "error_position_hist": error_position_hist,
        "get_run_diagnosis": get_run_diagnosis,
        "inspect_error_step": inspect_error_step,
        "compare_models_on_app": compare_models_on_app,
        "compare_models_on_task": compare_models_on_task,
        "model_error_taxonomy": model_error_taxonomy,
        "hard_cases": hard_cases,
        "llm_run_judge": llm_run_judge,
        "llm_model_meta_review": llm_model_meta_review,
    }


def tool_schema():
    return [
        {
            "type": "function",
            "function": {
                "name": "list_runs",
                "description": "列出所有已索引的运行（简要信息）。",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_run",
                "description": "获取指定 run_id 的运行摘要。",
                "parameters": {
                    "type": "object",
                    "properties": {"run_id": {"type": "string"}},
                    "required": ["run_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sample_failures",
                "description": "抽样返回某模型的失败案例。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "limit": {"type": "integer", "default": 3},
                    },
                    "required": ["model"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sample_successes",
                "description": "抽样返回某模型的成功案例。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "limit": {"type": "integer", "default": 3},
                    },
                    "required": ["model"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_steps",
                "description": "获取某个 run 的指定区间步骤明细。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "start": {"type": "integer", "default": 0},
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["run_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_task_context",
                "description": "返回 run 的任务上下文（app、task、instruction 等）。",
                "parameters": {
                    "type": "object",
                    "properties": {"run_id": {"type": "string"}},
                    "required": ["run_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_step_image",
                "description": "获取指定步骤的截图路径或压缩后的 base64。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "step": {"type": "integer"},
                        "which": {"type": "string", "enum": ["prev", "new"], "default": "prev"},
                        "embed": {"type": "boolean", "default": False},
                        "max_size": {"type": "integer", "default": 512},
                    },
                    "required": ["run_id", "step"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stats_by_task",
                "description": "按任务类型聚合某模型的成功率与平均步数。",
                "parameters": {
                    "type": "object",
                    "properties": {"model": {"type": "string"}},
                    "required": ["model"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "op_distribution",
                "description": "统计某模型的动作分布（点击/滑动/输入占比）。",
                "parameters": {
                    "type": "object",
                    "properties": {"model": {"type": "string"}},
                    "required": ["model"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "last_fail_reason",
                "description": "查看模型失败案例的最后一步动作与状态，定位常见触发点。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "limit": {"type": "integer", "default": 5},
                    },
                    "required": ["model"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "failure_slice",
                "description": "查看某个失败 run 的尾部若干步骤，用于定位错误位置。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "tail": {"type": "integer", "default": 5},
                    },
                    "required": ["run_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "loop_signals",
                "description": "检测一个 run 是否出现循环/卡住（看状态标签与图片是否重复）。",
                "parameters": {
                    "type": "object",
                    "properties": {"run_id": {"type": "string"}},
                    "required": ["run_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "error_position_hist",
                "description": "统计失败 run 最后标签的分布，推测常见出错位置。",
                "parameters": {
                    "type": "object",
                    "properties": {"model": {"type": "string"}},
                    "required": ["model"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_run_diagnosis",
                "description": "返回单条轨迹的结构化诊断：首个错误步、关键错误、原因候选。",
                "parameters": {
                    "type": "object",
                    "properties": {"run_id": {"type": "string"}},
                    "required": ["run_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_error_step",
                "description": "细看单个步骤：动作、状态变化、截图差异分值与候选原因。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "step": {"type": "integer"},
                    },
                    "required": ["run_id", "step"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_models_on_app",
                "description": "按指定 app 对比两模型表现，并补充 app 类型。",
                "parameters": {
                    "type": "object",
                    "properties": {"app": {"type": "string"}},
                    "required": ["app"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_models_on_task",
                "description": "按指定 task 对比两模型表现。",
                "parameters": {
                    "type": "object",
                    "properties": {"task": {"type": "string"}},
                    "required": ["task"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "model_error_taxonomy",
                "description": "汇总某模型失败原因分类计数。",
                "parameters": {
                    "type": "object",
                    "properties": {"model": {"type": "string"}},
                    "required": ["model"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "hard_cases",
                "description": "返回最难失败样本（按卡住强度排序），含 app/task 类型与原因。",
                "parameters": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "default": 10}},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "llm_run_judge",
                "description": "对单条 run 进行 LLM 复核，输出错误阶段、根因、思考/动作/终止问题。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "with_images": {"type": "boolean", "default": False},
                    },
                    "required": ["run_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "llm_model_meta_review",
                "description": "对某个模型做 LLM 元评审，归纳思考过程问题与改进建议。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "limit": {"type": "integer", "default": 6},
                    },
                    "required": ["model"],
                },
            },
        },
    ]


# ----------------------------- chat loop -----------------------------

SYSTEM_PROMPT = """

你是一名 GUI 智能体的评测助手。
- 不仅关注成功率和步数，还要结合任务类型、应用场景、出错位置与原因进行分析。
- 结合提供的统计数据，分析模型的操作风格、优势、薄弱环节与常见失败模式。
- 当需要细节（具体运行、步骤、截图、任务文本）时调用工具；若无需细节直接依据聚合数据给出结论。
- 优先使用工具：model_error_taxonomy / hard_cases / llm_model_meta_review / get_run_diagnosis / llm_run_judge / inspect_error_step / compare_models_on_app。
- 输出请使用中文，包含：
  1) 总体表现（含 app 类型与任务意图维度）
  2) 模型特点对比
  3) 错误发生位置与证据（步骤、标签、截图路径）
  4) 错误原因假设（区分规则诊断与 LLM 复核结论）
  5) 具体改进建议（策略、提示词、动作约束、终止条件）

"""


def run_chat(
    client: Any,
    run_index: Dict[str, RunSummary],
    aggregates: Dict[str, Dict[str, float]],
    dimensional_aggregates: Dict[str, Dict[str, dict]],
    model_name: str,
    judge_model: str,
    logger: logging.Logger,
):
    tools = build_tools(run_index, llm_client=client, judge_model=judge_model)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "aggregates": aggregates,
                    "dimensional_aggregates": dimensional_aggregates,
                    "任务": "请对 UI_TARS 与 MobiMind 给出深入评测，不止看成功率，要明确错误出现在哪类任务、哪类 app、哪个步骤附近以及可能原因。",
                    "建议流程": [
                        "先调用 model_error_taxonomy 与 hard_cases 看失败全貌",
                        "再调用 llm_model_meta_review 做模型级复核",
                        "然后调用 get_run_diagnosis/llm_run_judge/inspect_error_step 抽样核验",
                        "必要时用 get_step_image 给出截图证据路径",
                        "最后给出按优先级排序的改进建议",
                    ],
                },
                ensure_ascii=False,
            ),
        },
    ]

    round_idx = 0
    while True:
        round_idx += 1
        logger.info(f"[chat] round={round_idx} 请求模型")
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tool_schema(),
            tool_choice="auto",
            temperature=0,
        )
        logger.info(f"[chat] round={round_idx} 模型响应耗时={time.perf_counter() - t0:.2f}s")
        choice = resp.choices[0]
        msg = choice.message
        if msg.tool_calls:
            for call in msg.tool_calls:
                fn = tools.get(call.function.name)
                if not fn:
                    logger.warning(f"[tool] 未知工具: {call.function.name}")
                    continue
                args = json.loads(call.function.arguments or "{}")
                logger.info(f"[tool] 调用 {call.function.name} args={json.dumps(args, ensure_ascii=False)}")
                t1 = time.perf_counter()
                result = fn(**args)
                logger.info(
                    f"[tool] 完成 {call.function.name} 耗时={time.perf_counter() - t1:.2f}s, "
                    f"result_type={type(result).__name__}"
                )
                messages.append(
                    {"role": "assistant", "tool_calls": [call]}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.function.name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            continue
        final_text = msg.content or ""
        print(final_text)
        logger.info("[chat] 获得最终文本输出")
        return final_text


# ----------------------------- main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui_dir", default="/home/fff/mobibench/MobiBench/agents/UI_TARS/runs/20260316")
    parser.add_argument("--mm_dir", default="/home/fff/mobibench/MobiBench/agents/MobiMind/data/20260315")
    parser.add_argument("--api_key", default='key')
    parser.add_argument("--base_url", default='http://ipxx.chat.gpt:3006/v1')
    parser.add_argument("--model", default='anthropic/claude-sonnet-4.6')
    parser.add_argument("--judge_model", default='google/gemini-3-flash-preview')
    parser.add_argument("--output_dir", default="/home/fff/mobibench/MobiBench/eval_outputs")
    parser.add_argument("--diag_log_file", default="")
    parser.add_argument("--markdown_out", default="")
    parser.add_argument("--audio_out", default="")
    parser.add_argument("--enable_tts", action="store_true")
    args = parser.parse_args()

    ts = _now_ts()
    output_dir = args.output_dir
    if not args.diag_log_file:
        args.diag_log_file = os.path.join(output_dir, f"diagnosis_{ts}.log")
    if not args.markdown_out:
        args.markdown_out = os.path.join(output_dir, f"report_{ts}.md")
    if not args.audio_out:
        args.audio_out = os.path.join(output_dir, f"report_{ts}.wav")

    logger = setup_logger(args.diag_log_file)
    logger.info(f"[start] output_dir={output_dir}")
    logger.info(f"[start] diag_log_file={args.diag_log_file}")

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("缺少 openai 依赖，请先安装: pip install openai") from e

    t_scan = time.perf_counter()
    ui_runs = scan_runs(args.ui_dir, "UI_TARS")
    mm_runs = scan_runs(args.mm_dir, "MobiMind")
    logger.info(f"[scan] 完成目录扫描，耗时={time.perf_counter() - t_scan:.2f}s")
    run_index = {**ui_runs, **mm_runs}
    trace_cache = {rid: load_trace(rid, run_index) for rid in run_index}

    aggregates = {
        "UI_TARS": aggregate(list(ui_runs.values())),
        "MobiMind": aggregate(list(mm_runs.values())),
    }
    dimensional_aggregates = {
        "UI_TARS": aggregate_with_dimensions(list(ui_runs.values()), trace_cache),
        "MobiMind": aggregate_with_dimensions(list(mm_runs.values()), trace_cache),
    }

    client = OpenAI(api_key=args.api_key, base_url=args.base_url) if args.base_url else OpenAI(api_key=args.api_key)

    logger.info(f"[info] indexed runs: UI_TARS={len(ui_runs)}, MobiMind={len(mm_runs)}")
    logger.info(f"[info] aggregates: {json.dumps(aggregates, ensure_ascii=False)}")
    logger.info("[info] 已启用深度诊断维度: by_app/by_app_type/by_task/by_intent")
    print(f"[info] indexed runs: UI_TARS={len(ui_runs)}, MobiMind={len(mm_runs)}")
    print(f"[info] aggregates: {json.dumps(aggregates, ensure_ascii=False)}")
    print("[info] 已启用深度诊断维度: by_app/by_app_type/by_task/by_intent")

    judge_model = args.judge_model or args.model
    final_text = run_chat(client, run_index, aggregates, dimensional_aggregates, args.model, judge_model, logger)

    save_markdown_report(args.markdown_out, aggregates, dimensional_aggregates, final_text)
    logger.info(f"[output] markdown 已保存: {args.markdown_out}")
    print(f"[info] markdown report saved to: {args.markdown_out}")

    if args.enable_tts:
        ok = save_tts_audio(final_text, args.audio_out, logger)
        if ok:
            print(f"[info] audio report saved to: {args.audio_out}")
        else:
            print("[warn] tts failed, skipped audio output")


if __name__ == "__main__":
    main()
