# -*- coding: utf-8 -*-
"""
multi_vis.py —— FSM 可视化（标签级）
- 不显示边上的「click × 次数」；默认不显示任何边标签
- 支持将指定标签按从左到右固定位置（例如 --order 1,2,3 或 1,2,3,done）
- 支持 live 模式（直接解析目录）与 json 模式（从 fsm_traces.json 读取）
"""
import os
import re
import sys
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ---------- 中文字体与负号 ----------
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ---------- 兼容导入 AppFSM ----------
def _try_import_appfsm():
    try:
        # 从工程根运行：python -m MobiFlow.static_bench.multi_vis ...
        from static_bench.multi_trace_to_one import AppFSM  # type: ignore
        return AppFSM
    except Exception:
        # 退而求其次：把脚本上两级加入 sys.path（适配单文件放置）
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if base not in sys.path:
            sys.path.insert(0, base)
        try:
            from multi_trace_to_one import AppFSM  # type: ignore
            return AppFSM
        except Exception:
            return None


# ---------- 轻量数据结构（当不依赖项目类时） ----------
class _StateNS:
    __slots__ = ("img_path", "map_info", "cluster_class")
    def __init__(self, img_path, map_info=None, cluster_class=None):
        self.img_path = img_path
        self.map_info = map_info or {"click": {}, "swipe": {}, "input": {}, "wait": {}}
        self.cluster_class = cluster_class

class _TraceNS:
    __slots__ = ("states", "actions")
    def __init__(self, states=None, actions=None):
        self.states = states or []
        self.actions = actions or []


# ---------- JSON 键还原（"TUPLE:(a, b, c, d)" -> (a,b,c,d)） ----------
_TUPLE_PREFIX = "TUPLE:("

def _parse_tuple_key(k):
    if isinstance(k, str) and k.startswith(_TUPLE_PREFIX) and k.endswith(")"):
        body = k[len(_TUPLE_PREFIX):-1]
        return tuple(int(x.strip()) for x in body.split(","))
    return k

def _restore_map_info_keys_in_place(state_dict):
    """把 JSON 里的字符串键还原为元组键"""
    for cat in ("click", "swipe", "input", "wait"):
        if cat in state_dict.get("map_info", {}):
            mi = state_dict["map_info"][cat]
            if isinstance(mi, dict):
                newd = {}
                for k, v in mi.items():
                    newd[_parse_tuple_key(k)] = v
                state_dict["map_info"][cat] = newd


# ---------- Loader ----------
def load_traces_from_json(json_path: str):
    """从 fsm_traces.json 读回简单对象（无项目依赖）"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    traces = []
    for t in data.get("traces", []):
        st_objs = []
        for s in t.get("states", []):
            _restore_map_info_keys_in_place(s)
            st_objs.append(_StateNS(
                img_path=s.get("img_path"),
                map_info=s.get("map_info", {}),
                cluster_class=s.get("cluster_class")
            ))
        traces.append(_TraceNS(states=st_objs, actions=[]))  # json 通常不带 actions
    return traces

def load_traces_live(data_path: str, app: str, task: str):
    """用 AppFSM 即时解析目录"""
    AppFSM = _try_import_appfsm()
    if AppFSM is None:
        raise RuntimeError("无法导入 AppFSM。请从工程根运行，或检查路径/包。")
    fsm = AppFSM(app, task, data_path)
    # fsm.traces 里应包含 states 与（我们在 init 时挂上的）actions
    return fsm.traces


# ---------- 构图（标签级） ----------
def _label_of(state) -> str:
    """有标签返回标签；无标签返回 '目录名/文件名' 保证唯一且短"""
    lbl = getattr(state, "cluster_class", None)
    if lbl:
        return str(lbl)
    p = Path(getattr(state, "img_path"))
    return f"{p.parent.name}/{p.stem}"

def build_label_multigraph_from_traces(trace_links):
    """
    输出: MultiDiGraph
      - 节点：label（cluster_class 或 目录/文件）
      - 边： (src_label) -[action]-> (tgt_label)，累积 weight
    源：优先使用 actions；若无 actions，则从 map_info 恢复。
    """
    G = nx.MultiDiGraph()

    # 先收集所有 img_path -> label 的映射（供 map_info 模式使用）
    path2label = {}
    for tl in trace_links:
        for s in getattr(tl, "states", []) or []:
            path2label[getattr(s, "img_path")] = _label_of(s)

    for tl in trace_links:
        states = getattr(tl, "states", []) or []
        actions = getattr(tl, "actions", []) or []
        if not states:
            continue

        # 确保所有节点存在
        for s in states:
            G.add_node(_label_of(s))

        # A) 优先：按 actions 时间轴连边
        if actions and len(states) >= 2:
            steps = min(len(actions), len(states) - 1)
            for i in range(steps):
                src = _label_of(states[i])
                tgt = _label_of(states[i + 1])
                a = getattr(actions[i], "act_type", None) or getattr(actions[i], "type", "unknown")
                _add_edge_with_count(G, src, tgt, a)
            continue  # 本条 trace 已处理

        # B) 备选：从 map_info 中恢复（聚合后的并集可能跨轨迹）
        for s in states:
            src = _label_of(s)
            mi = getattr(s, "map_info", {}) or {}
            for act_type in ("click", "swipe", "input", "wait"):
                for _, dst_path in (mi.get(act_type, {}) or {}).items():
                    tgt = path2label.get(dst_path)
                    if tgt is None:
                        continue
                    _add_edge_with_count(G, src, tgt, act_type)

    return G

def _add_edge_with_count(G: nx.MultiDiGraph, src: str, tgt: str, act_type: str):
    key = act_type or "unknown"
    if G.has_edge(src, tgt, key=key):
        G[src][tgt][key]["weight"] += 1
    else:
        G.add_edge(src, tgt, key=key, action=key, weight=1)


# ---------- 可视化 ----------
def visualize_label_multigraph(trace_links, layout="spring", save_path: str = None,
                               ordered_labels=None, show_edge_labels: bool = False):
    """
    ordered_labels: 例如 ["1","2","3"]，将这些标签按从左到右固定
    show_edge_labels: True 时仅显示动作名称（不显示“×次数”）；默认 False
    """
    G = build_label_multigraph_from_traces(trace_links)
    if G.number_of_nodes() == 0:
        print("图为空：没有可用的状态/转移。")
        return

    # —— 锚点：把 ordered_labels 固定在一条直线左→右 —— #
    anchors = {}
    if ordered_labels:
        # x 间距可调，这里用 2.0；y 放在 0 轴
        for i, lbl in enumerate(ordered_labels):
            if lbl in G.nodes:
                anchors[lbl] = np.array([i * 2.0, 0.0])

    # 布局
    if layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
        if anchors:  # 覆盖锚点坐标
            for k, v in anchors.items():
                pos[k] = v
    elif layout == "circular":
        pos = nx.circular_layout(G)
        if anchors:
            for k, v in anchors.items():
                pos[k] = v
    else:
        # spring：可以真正“固定”锚点
        pos = nx.spring_layout(
            G,
            k=1/np.sqrt(max(1, G.number_of_nodes())),
            iterations=300,
            seed=42,
            pos=anchors if anchors else None,
            fixed=list(anchors.keys()) if anchors else None
        )

    # 节点颜色
    nodes = list(G.nodes())
    node_colors = plt.cm.Set3(np.linspace(0, 1, len(nodes)))
    node_color_map = {n: node_colors[i] for i, n in enumerate(nodes)}
    node_color_list = [node_color_map[n] for n in nodes]

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=900,
                           node_color=node_color_list, edgecolors="black", alpha=0.95)
    nx.draw_networkx_labels(G, pos, font_size=9)

    # 边按动作类型分组并画成弧线，减少重叠（无边标签计数）
    action2color = {"click": "red", "swipe": "green", "input": "orange", "wait": "gray"}
    default_edge_color = "black"
    def arc_for_action(a: str) -> float:
        table = {"click": 0.15, "swipe": -0.15, "input": 0.25, "wait": -0.25}
        return table.get(a, 0.10)

    edges_by_action = defaultdict(list)
    for u, v, k, data in G.edges(keys=True, data=True):
        a = data.get("action", k)
        edges_by_action[a].append((u, v, k, data))

    for a, edges in edges_by_action.items():
        color = action2color.get(a, default_edge_color)
        for (u, v, k, d) in edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)],
                edge_color=color, arrows=True, arrowsize=16, width=1.8, alpha=0.8,
                connectionstyle=f"arc3,rad={arc_for_action(a)}"
            )

    # —— 边标签（默认不画；仅在需要时显示“动作名”，不含次数） —— #
    if show_edge_labels:
        edge_labels = {}
        for u, v, k, data in G.edges(keys=True, data=True):
            a = data.get("action", k)
            key_uv = (u, v)
            text = f"{a}"
            edge_labels[key_uv] = (edge_labels.get(key_uv, "") + ("\n" if key_uv in edge_labels else "") + text)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5)

    # 图例
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=c, label=act) for act, c in action2color.items()]
    plt.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)

    plt.title("FSM (Label-level, merged from multiple traces)")
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        out = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[Saved] {out}")
    else:
        plt.show()

    # 控制台统计（无计数标签，但仍可作为参考）
    print(f"节点数: {G.number_of_nodes()} | 边数(含平行多边): {G.number_of_edges()}")


# ---------- CLI ----------
def _build_argparser():
    import argparse
    p = argparse.ArgumentParser(description="FSM 可视化（标签级）")
    p.add_argument("--mode", choices=["live", "json"], default="live",
                   help="数据来源：live=即时解析；json=从 fsm_traces.json 读")
    p.add_argument("--data_path", type=str, default=r"/Users/fff/Desktop/mobiagent/Mobibench/data",
                   help="数据根目录（live 模式必填）")
    p.add_argument("--app", type=str, default="美团", help="应用名（live 模式）")
    p.add_argument("--task", type=str, default="type1", help="任务名（live 模式）")
    p.add_argument("--json_path", type=str, default=None,
                   help="fsm_traces.json 路径（json 模式可不填，自动用 data_path/fsm/app/task/fsm_traces.json）")
    p.add_argument("--layout", choices=["spring", "kamada", "circular"], default="spring",
                   help="图布局（推荐 spring，便于固定锚点）")
    p.add_argument("--save", type=str, default=None, help="保存图片到此路径（留空则弹窗显示）")
    p.add_argument("--order", type=str, default=None,
                   help="按从左到右固定的标签顺序，例如 1,2,3 或 1,2,3,done")
    p.add_argument("--edge-labels", action="store_true",
                   help="显示边上的动作名（不含次数）")
    return p


def main():
    args = _build_argparser().parse_args()

    # 加载数据
    if args.mode == "live":
        traces = load_traces_live(args.data_path, args.app, args.task)
    else:
        if args.json_path:
            jp = args.json_path
        else:
            jp = os.path.join(args.data_path, "fsm", args.app, args.task, "fsm_traces.json")
        if not os.path.isfile(jp):
            raise FileNotFoundError(f"找不到 json：{jp}")
        traces = load_traces_from_json(jp)

    ordered = [s.strip() for s in args.order.split(",")] if args.order else None

    visualize_label_multigraph(
        traces,
        layout=args.layout,
        save_path=args.save,
        ordered_labels=ordered,
        show_edge_labels=args.edge_labels
    )


if __name__ == "__main__":
    main()
