# -*- coding: utf-8 -*-
"""
multi_vis_tagstep_with_boxes.py —— Tag 聚合 + 中间步骤展开 + 可选画 UI 框 的 FSM 可视化

- 所有 tag（包括 START / done）是全局共享节点：多条轨迹从不同路径汇聚到同一个 tag 点
- 其它非 tag 步骤（click / input / swipe ...）每一步是一个独立节点 + 一张截图
- 可选：在每张截图上叠加 OmniParser 检出的所有 UI 框（可交互区域）

使用方式示例：

python multi_vis_tagstep_with_boxes.py ^
  --data_path D:\cdl\code\MobiBench\MobiBench\collect\manual\data ^
  --app 携程 ^
  --task type9 ^
  --order START,search,input1,searchfill,date,allfill,target,hotel1,hotelf,done ^
  --edge-labels ^
  --show-boxes ^
  --save D:\cdl\code\MobiBench\runs\fsm_携程_type9_boxes.png
"""

import os
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# 这一行是关键：用你现有的检测代码来提取所有 bounding boxes
from MobiBench.utils.parse_omni import extract_all_bounds  # 提取 [left, top, right, bottom] 列表

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class TagStepFSMVisualizer:
    """
    Tag 聚合 + 中间步骤展开 的 FSM 可视化器，
    可选叠加每张截图中的所有 UI 检测框（可点击区域）。
    """

    def __init__(self, data_root: str, app: str, task: str, show_boxes: bool = False):
        self.data_root = data_root
        self.app = app
        self.task = task
        self.show_boxes = show_boxes

        self.traces = []          # load_runs 的结果
        self.G = None             # networkx MultiDiGraph
        self.pos = None           # 节点坐标
        self._box_cache = {}      # img_path -> bounds_list 缓存，避免重复跑检测

    # ---------- 1. 读取 actions.json + 截图 ----------

    def load_runs(self):
        """
        从 data_root/app/task/*/actions.json 读取多条轨迹，并收集截图路径。
        目录结构示例：
          data_root/app/task/1/actions.json + *.jpg
                              2/actions.json + *.jpg
                              ...
        """
        base = Path(self.data_root) / self.app / self.task
        traces = []
        IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

        if not base.exists():
            raise FileNotFoundError(base)

        for child in sorted(base.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            act_path = child / "actions.json"
            if not act_path.is_file():
                continue

            with open(act_path, "r", encoding="utf-8") as f:
                jd = json.load(f)
            acts = jd.get("actions", [])
            if not acts:
                continue

            imgs = sorted(
                [p for p in child.iterdir()
                 if p.is_file() and p.suffix.lower() in IMG_EXTS],
                key=lambda p: p.name
            )
            traces.append({"actions": acts, "images": [str(p) for p in imgs]})

        print(f"[INFO] 从 {base} 读取到 {len(traces)} 条轨迹(actions.json + images)")
        self.traces = traces

    # ---------- 2. 构图（tag 聚合，step 单独） ----------

    @staticmethod
    def _add_edge(G, src, tgt, act_type):
        key = act_type or "unknown"
        if G.has_edge(src, tgt, key=key):
            G[src][tgt][key]["weight"] += 1
        else:
            G.add_edge(src, tgt, key=key, action=key, weight=1)

    def build_graph_tag_step(self):
        """
        构建 MultiDiGraph：

        节点类型：
          - tag 节点： "START"、各个 label、"done"，全局唯一
          - step 节点：每条轨迹的每一个非 tag 步骤，节点名形如 "t0_s3"

        每个节点属性：
          - kind: "tag" 或 "step"
          - img_path: 对应截图路径
        """
        G = nx.MultiDiGraph()

        for ti, trace in enumerate(self.traces):
            acts = trace["actions"]
            imgs = trace["images"]
            if not acts or not imgs:
                continue

            # 全局 START 节点（只建一次），用第一条轨迹的第一张图
            if "START" not in G:
                G.add_node("START", kind="tag", img_path=imgs[0])

            prev_node = "START"
            last_act_type = None  # 最近一次非 tag 动作类型

            for si, a in enumerate(acts):
                t = a.get("type")
                img_idx = min(si, len(imgs) - 1)
                img_path = imgs[img_idx]

                if t == "tag":
                    label = str(a.get("label", "UNK"))
                    # 全局唯一 tag 节点
                    if label not in G:
                        G.add_node(label, kind="tag", img_path=img_path)
                    # 用最近一次非 tag 动作类型连接到 tag
                    edge_act = last_act_type or "tag"
                    self._add_edge(G, prev_node, label, edge_act)
                    prev_node = label
                    last_act_type = None

                elif t == "done":
                    label = "done"
                    if label not in G:
                        G.add_node(label, kind="tag", img_path=img_path)
                    edge_act = last_act_type or "done"
                    self._add_edge(G, prev_node, label, edge_act)
                    prev_node = label
                    last_act_type = None

                else:
                    # 非 tag 步骤：独立 step 节点
                    node_name = f"t{ti}_s{si}"
                    if node_name not in G:
                        G.add_node(node_name, kind="step", img_path=img_path)
                    edge_act = t or "unknown"
                    self._add_edge(G, prev_node, node_name, edge_act)
                    prev_node = node_name
                    last_act_type = t

        self.G = G

    # ---------- 3. spring 布局 + 固定 tag 顺序 ----------

    def compute_pos_with_anchors(self, ordered_labels=None, x_gap=10.0):
        """
        使用 spring_layout，但对 ordered_labels 中的 tag 加锚点，
        固定在一条水平线上：START, search, ..., done
        """
        G = self.G
        anchors = {}
        if ordered_labels:
            for i, lbl in enumerate(ordered_labels):
                if lbl in G.nodes:
                    anchors[lbl] = np.array([i * x_gap, 0.0])

        if anchors:
            pos = nx.spring_layout(
                G,
                k=1 / np.sqrt(max(1, G.number_of_nodes())),
                iterations=500,
                seed=42,
                pos=anchors,
                fixed=list(anchors.keys()),
            )
        else:
            pos = nx.spring_layout(
                G,
                k=1 / np.sqrt(max(1, G.number_of_nodes())),
                iterations=500,
                seed=42,
            )
        self.pos = pos

    @staticmethod
    def _node_seed(name: str) -> int:
        """简单确定性 hash，用于给 step 节点生成抖动偏移"""
        return sum(ord(c) for c in str(name))

    # ---------- 4. 叠加 UI 框 ----------

    def annotate_clickable_boxes(self, pil_img: Image.Image, img_path: str) -> Image.Image:
        """
        调用 parse_omni.extract_all_bounds，获取该截图中的所有 UI 元素框，
        并在图片上用绿色矩形画出这些框。

        返回：带框的 PIL.Image
        """
        # 缓存一下，避免同一张图重复跑检测
        if img_path in self._box_cache:
            bounds_list = self._box_cache[img_path]
        else:
            try:
                bounds_list = extract_all_bounds(img_path)
            except Exception as e:
                print(f"[WARN] 提取 UI 框失败 {img_path}: {e}")
                bounds_list = []
            self._box_cache[img_path] = bounds_list

        if not bounds_list:
            return pil_img

        draw = ImageDraw.Draw(pil_img)
        for (left, top, right, bottom) in bounds_list:
            draw.rectangle((left, top, right, bottom),
                           outline=(0, 255, 0),
                           width=3)
        return pil_img

    # ---------- 5. 画图 ----------

    def draw_graph(self, show_edge_labels=False, save_path=None):
        G = self.G
        pos = self.pos

        # ---- 整体放大坐标 + 给 step 节点加抖动，让节点尽量分散 ----
        scale_x, scale_y = 4.0, 8.0          # 全局放大倍数
        jitter_x, jitter_y = 1.5, 3.0        # step 节点额外抖动范围

        for n in pos:
            x, y = pos[n]
            kind = G.nodes[n].get("kind", "step")

            x *= scale_x
            y *= scale_y

            if kind == "step":
                s = self._node_seed(n)
                dx = ((s % 201) - 100) / 100.0 * jitter_x   # [-jitter_x, jitter_x]
                dy = (((s // 201) % 201) - 100) / 100.0 * jitter_y
                x += dx
                y += dy

            pos[n] = np.array([x, y])

        nodes = list(G.nodes())
        fig, ax = plt.subplots(figsize=(40, 15))

        # --- 画节点（tag 大图，step 小图），可选叠加 UI 框 ---
        for n in nodes:
            x, y = pos[n]
            kind = G.nodes[n].get("kind", "step")
            img_path = G.nodes[n].get("img_path", None)

            zoom = 0.26 if kind == "tag" else 0.16

            if img_path and os.path.isfile(img_path):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img.thumbnail((220, 220))

                    # 如果开启 show_boxes，就在节点截图上叠加检测框
                    if self.show_boxes:
                        img = self.annotate_clickable_boxes(img, img_path)

                    im = OffsetImage(np.asarray(img), zoom=zoom)
                    ab = AnnotationBbox(
                        im,
                        (x, y),
                        frameon=True,
                        pad=0.06 if kind == "tag" else 0.03,
                    )
                    ax.add_artist(ab)
                except Exception as e:
                    print(f"[WARN] 打开图片失败 {img_path}: {e}")
                    ax.scatter(
                        [x],
                        [y],
                        s=500 if kind == "tag" else 200,
                        c="lightgray",
                        edgecolors="black",
                        zorder=3,
                    )
            else:
                ax.scatter(
                    [x],
                    [y],
                    s=500 if kind == "tag" else 200,
                    c="lightgray",
                    edgecolors="black",
                    zorder=3,
                )

            # 只给 tag 写文字，避免 step 太密
            if kind == "tag":
                ax.text(
                    x,
                    y - 1.2,
                    str(n),
                    fontsize=10,
                    ha="center",
                    va="top",
                    zorder=4,
                )

        # --- 画边 ---
        action2color = {
            "click": "red",
            "swipe": "green",
            "input": "orange",
            "wait": "gray",
            "tag": "blue",
            "done": "purple",
        }
        default_edge_color = "black"

        def arc_for_action(a):
            table = {
                "click": 0.15,
                "swipe": -0.15,
                "input": 0.25,
                "wait": -0.25,
                "tag": 0.20,
                "done": -0.20,
            }
            return table.get(a, 0.10)

        edges_by_action = defaultdict(list)
        for u, v, k, data in G.edges(keys=True, data=True):
            a = data.get("action", k)
            edges_by_action[a].append((u, v, k, data))

        for a, edges in edges_by_action.items():
            color = action2color.get(a, default_edge_color)
            for (u, v, k, d) in edges:
                w = d.get("weight", 1)
                width = 1.2 + 0.5 * np.log1p(w)
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=[(u, v)],
                    edge_color=color,
                    arrows=True,
                    arrowsize=14,
                    width=width,
                    alpha=0.85,
                    connectionstyle=f"arc3,rad={arc_for_action(a)}",
                    ax=ax,
                )

        # --- 边标签：不显示 click，只显示 input / swipe / done 等 ---
        if show_edge_labels:
            edge_labels = {}
            for u, v, k, data in G.edges(keys=True, data=True):
                a = data.get("action", k)
                if a == "click":
                    continue  # click 的边不画文字

                w = data.get("weight", 1)
                key_uv = (u, v)
                text = f"{a}×{w}"
                edge_labels[key_uv] = (
                    edge_labels.get(key_uv, "")
                    + ("\n" if key_uv in edge_labels else "")
                    + text
                )
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_labels,
                font_size=8,
                label_pos=0.5,
                ax=ax,
            )

        # 图例
        import matplotlib.patches as mpatches

        legend_patches = [
            mpatches.Patch(color="red", label="click"),
            mpatches.Patch(color="green", label="swipe"),
            mpatches.Patch(color="orange", label="input"),
            mpatches.Patch(color="gray", label="wait"),
            mpatches.Patch(color="blue", label="tag"),
            mpatches.Patch(color="purple", label="done"),
        ]
        ax.legend(
            handles=legend_patches,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )

        ax.set_title("Tag-aggregated FSM with per-step nodes")
        ax.axis("off")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved {save_path}")
        else:
            plt.show()

        print(
            f"节点数: {G.number_of_nodes()} | 边数(含平行多边): {G.number_of_edges()}"
        )

    # ---------- 6. 一键跑完 ----------

    def run(self, ordered_labels=None, show_edge_labels=False, save_path=None):
        self.load_runs()
        self.build_graph_tag_step()
        self.compute_pos_with_anchors(ordered_labels=ordered_labels, x_gap=10.0)
        self.draw_graph(show_edge_labels=show_edge_labels, save_path=save_path)


# ---------- 7. CLI ----------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tag 聚合 + 中间步骤展开 的 FSM 可视化（基于 actions.json，支持叠加 UI 框）"
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="数据根目录，例如 D:/.../collect/manual/data",
    )
    parser.add_argument(
        "--app",
        required=True,
        help="应用名，例如 携程",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="任务名，例如 type9",
    )
    parser.add_argument(
        "--order",
        default=None,
        help=(
            "tag 顺序，用于固定 START→...→done 的位置，例如 "
            "START,search,input1,searchfill,date,allfill,target,hotel1,hotelf,done"
        ),
    )
    parser.add_argument(
        "--edge-labels",
        action="store_true",
        help="是否显示边上的动作名和次数（click 会被自动忽略）",
    )
    parser.add_argument(
        "--show-boxes",
        action="store_true",
        help="在每张截图上叠加 OmniParser 检出的所有 UI 框",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="保存图片路径",
    )

    args = parser.parse_args()

    ordered = [s.strip() for s in args.order.split(",")] if args.order else None

    vis = TagStepFSMVisualizer(
        data_root=args.data_path,
        app=args.app,
        task=args.task,
        show_boxes=args.show_boxes,
    )
    vis.run(
        ordered_labels=ordered,
        show_edge_labels=args.edge_labels,
        save_path=args.save,
    )
