# -*- coding: utf-8 -*-
"""
multi_vis_tagstep_with_boxes_auto.py —— 自动优化布局的Tag聚合FSM可视化

改进点：
1. 自动计算最佳tag顺序（基于最长路径和连接密度）
2. 优化的美观布局（层次布局+力导向优化）
3. 智能颜色方案和视觉效果
4. 自适应节点大小和间距
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import heapq

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyBboxPatch

# 检测UI框的代码
try:
    from MobiBench.utils.parse_omni import extract_all_bounds
except ImportError:
    print("[WARN] OmniParser模块未找到，将无法显示UI框")
    extract_all_bounds = lambda x: []

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150


class AutoLayoutFSMVisualizer:
    """
    自动布局的FSM可视化器，无需手动指定tag顺序
    优化的视觉效果，更美观的图形展示
    """

    def __init__(self, data_root: str, app: str, task: str, show_boxes: bool = False):
        self.data_root = data_root
        self.app = app
        self.task = task
        self.show_boxes = show_boxes

        self.traces = []
        self.G = nx.MultiDiGraph()
        self.pos = {}
        self._box_cache = {}
        
        # 美观度参数
        self.layout_config = {
            'tag_spacing': 15.0,       # tag节点间距
            'step_spacing': 3.0,       # step节点间距基数
            'layer_gap': 20.0,         # 层次间距
            'tag_radius': 1.0,         # tag节点抖动半径
            'step_radius': 5.0,        # step节点抖动半径
            'min_node_distance': 2.0,  # 最小节点间距
        }

    # ---------- 1. 数据加载 ----------

    def load_runs(self):
        """加载所有轨迹数据"""
        base = Path(self.data_root) / self.app / self.task
        traces = []
        IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

        if not base.exists():
            raise FileNotFoundError(f"路径不存在: {base}")

        run_dirs = sorted([d for d in base.iterdir() if d.is_dir()], 
                         key=lambda x: x.name)
        
        for run_dir in run_dirs:
            act_path = run_dir / "actions.json"
            if not act_path.is_file():
                continue

            try:
                with open(act_path, "r", encoding="utf-8") as f:
                    jd = json.load(f)
                acts = jd.get("actions", [])
                if not acts:
                    continue

                # 收集所有截图
                imgs = sorted(
                    [p for p in run_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in IMG_EXTS],
                    key=lambda p: p.name
                )
                
                if imgs:
                    traces.append({
                        "actions": acts, 
                        "images": [str(p) for p in imgs],
                        "run_id": run_dir.name
                    })
                    
            except Exception as e:
                print(f"[WARN] 加载轨迹失败 {run_dir}: {e}")

        print(f"[INFO] 从 {base} 读取到 {len(traces)} 条有效轨迹")
        self.traces = traces
        return len(traces) > 0

    # ---------- 2. 自动计算tag顺序 ----------
    def _compute_tag_order_automatically(self) -> List[str]:
    
        if not self.G or self.G.number_of_nodes() == 0:
            return ["START", "done"]
        
        # 提取所有tag节点
        tag_nodes = [n for n in self.G.nodes() 
                    if self.G.nodes[n].get('kind') == 'tag']
        
        # 确保包含START和done节点
        start_present = "START" in tag_nodes
        done_present = "done" in tag_nodes
        
        # 如果只有START和done，直接返回
        if len(tag_nodes) <= 2 and start_present and done_present:
            return ["START", "done"]
        
        # 构建简化的tag图（忽略step节点）
        tag_G = nx.DiGraph()
        for tag in tag_nodes:
            tag_G.add_node(tag)
        
        # 添加tag之间的连接（统计权重）
        for u, v, data in self.G.edges(data=True):
            if self.G.nodes[u].get('kind') == 'tag' and self.G.nodes[v].get('kind') == 'tag':
                weight = data.get('weight', 1)
                if tag_G.has_edge(u, v):
                    tag_G[u][v]['weight'] += weight
                else:
                    tag_G.add_edge(u, v, weight=weight)
        
        # 如果没有连接或只有很少的连接，使用智能排序
        if tag_G.number_of_edges() == 0 or len(tag_nodes) <= 3:
            # 首先确保START在最前，done在最后
            other_tags = [tag for tag in tag_nodes if tag not in ["START", "done"]]
            
            # 如果只有一个其他tag，直接放中间
            if len(other_tags) == 1:
                return ["START", other_tags[0], "done"]
            
            # 如果有多个其他tag，尝试找到最合理的顺序
            if tag_G.number_of_edges() > 0:
                # 使用节点的入度和出度来排序
                tag_scores = {}
                for tag in other_tags:
                    in_deg = tag_G.in_degree(tag, weight='weight')
                    out_deg = tag_G.out_degree(tag, weight='weight')
                    # 更靠近START的节点应该在前，靠近done的节点应该在后
                    tag_scores[tag] = out_deg - in_deg  # 正数表示更靠近START
                
                # 按分数排序，分数高的（靠近START）在前
                sorted_other_tags = sorted(other_tags, key=lambda x: tag_scores.get(x, 0), reverse=True)
                return ["START"] + sorted_other_tags + ["done"]
            else:
                # 完全没连接，使用字母顺序
                sorted_other_tags = sorted(other_tags)
                return ["START"] + sorted_other_tags + ["done"]
        
        try:
            # 方法1: 尝试使用关键路径（最长路径）
            if start_present and done_present:
                try:
                    # 找到从START到done的最长路径
                    longest_path = nx.dag_longest_path(tag_G, weight='weight')
                    if len(longest_path) >= 2 and longest_path[0] == "START" and longest_path[-1] == "done":
                        # 获取关键路径上的节点
                        critical_nodes = longest_path
                        
                        # 处理不在关键路径上的节点
                        non_critical = [n for n in tag_nodes if n not in critical_nodes]
                        
                        if non_critical:
                            # 为不在关键路径的节点找到最佳插入位置
                            final_order = []
                            for i, node in enumerate(critical_nodes):
                                final_order.append(node)
                                
                                # 在当前节点后插入与之连接的节点
                                if i < len(critical_nodes) - 1:
                                    current = critical_nodes[i]
                                    next_node = critical_nodes[i + 1]
                                    
                                    # 找到连接当前节点和下一个节点的中间节点
                                    connected_nodes = []
                                    for n in non_critical:
                                        # 检查是否有从当前到n，再从n到next_node的连接
                                        if (tag_G.has_edge(current, n) or tag_G.has_edge(n, next_node)):
                                            connected_nodes.append(n)
                                    
                                    # 按连接强度排序
                                    connected_nodes.sort(
                                        key=lambda x: tag_G.get_edge_data(current, x, {}).get('weight', 0) + 
                                                    tag_G.get_edge_data(x, next_node, {}).get('weight', 0),
                                        reverse=True
                                    )
                                    
                                    final_order.extend(connected_nodes)
                                    # 从non_critical中移除已处理的节点
                                    non_critical = [n for n in non_critical if n not in connected_nodes]
                            
                            # 如果还有剩余的节点，放在合适的位置
                            if non_critical:
                                # 放在done之前
                                final_order = final_order[:-1] + non_critical + ["done"]
                            
                            return final_order
                        else:
                            return critical_nodes
                except:
                    pass  # 如果最长路径失败，尝试其他方法
        except:
            pass  # 忽略异常，使用备用方案
        
        try:
            # 方法2: 使用拓扑排序（如果是有向无环图）
            order = list(nx.topological_sort(tag_G))
            
            # 确保START在最前，done在最后
            final_order = []
            
            # 添加START（如果存在）
            if "START" in order:
                final_order.append("START")
                order.remove("START")
            
            # 添加中间节点
            final_order.extend([n for n in order if n != "done"])
            
            # 添加done（如果存在）
            if "done" in order:
                final_order.append("done")
            elif "done" in tag_nodes:  # 如果done不在order中但存在于图中
                final_order.append("done")
            
            return final_order
        except nx.NetworkXUnfeasible:
            # 有环图，使用其他方法
            pass
        
        try:
            # 方法3: 使用PageRank算法计算重要性
            pagerank = nx.pagerank(tag_G, weight='weight')
            
            # 获取所有节点，排除START和done
            other_tags = [tag for tag in tag_nodes if tag not in ["START", "done"]]
            
            # 按PageRank值排序
            sorted_tags = sorted(other_tags, key=lambda x: pagerank.get(x, 0), reverse=True)
            
            # 构建最终顺序：START + 排序的其他节点 + done
            final_order = ["START"] + sorted_tags
            
            # 确保done在最后（如果存在）
            if "done" in tag_nodes:
                final_order.append("done")
            
            return final_order
        except:
            # 方法4: 基于连接数排序（最可靠的备用方案）
            # 首先确保START在最前，done在最后
            other_tags = [tag for tag in tag_nodes if tag not in ["START", "done"]]
            
            if not other_tags:
                return ["START", "done"] if "done" in tag_nodes else ["START"]
            
            # 计算每个节点的总连接权重
            tag_scores = {}
            for tag in other_tags:
                total_weight = 0
                # 入边权重
                for _, _, data in tag_G.in_edges(tag, data=True):
                    total_weight += data.get('weight', 0)
                # 出边权重
                for _, _, data in tag_G.out_edges(tag, data=True):
                    total_weight += data.get('weight', 0)
                tag_scores[tag] = total_weight
            
            # 按总权重排序
            sorted_other_tags = sorted(other_tags, key=lambda x: tag_scores.get(x, 0), reverse=True)
            
            final_order = ["START"] + sorted_other_tags
            if "done" in tag_nodes:
                final_order.append("done")
            
            return final_order

    # ---------- 3. 构建图 ----------

    @staticmethod
    def _add_edge(G, src, tgt, act_type):
        """添加边并记录权重"""
        key = act_type or "unknown"
        if G.has_edge(src, tgt, key=key):
            G[src][tgt][key]["weight"] += 1
        else:
            G.add_edge(src, tgt, key=key, action=key, weight=1)

    def build_graph(self):
        """构建完整的FSM图"""
        G = nx.MultiDiGraph()
        
        for ti, trace in enumerate(self.traces):
            acts = trace["actions"]
            imgs = trace["images"]
            if not acts or not imgs:
                continue

            # 添加START节点
            if "START" not in G:
                G.add_node("START", kind="tag", img_path=imgs[0], 
                          run_id=trace["run_id"], trace_idx=ti)

            prev_node = "START"
            last_non_tag_action = None

            for si, action in enumerate(acts):
                act_type = action.get("type", "")
                img_idx = min(si, len(imgs) - 1)
                img_path = imgs[img_idx]

                if act_type == "tag":
                    label = str(action.get("label", "UNK"))
                    if label not in G:
                        G.add_node(label, kind="tag", img_path=img_path,
                                  run_id=trace["run_id"], trace_idx=ti)
                    
                    # 使用最近的非tag动作类型作为边标签
                    edge_label = last_non_tag_action or "tag"
                    self._add_edge(G, prev_node, label, edge_label)
                    prev_node = label
                    last_non_tag_action = None

                elif act_type == "done":
                    label = "done"
                    if label not in G:
                        G.add_node(label, kind="tag", img_path=img_path,
                                  run_id=trace["run_id"], trace_idx=ti)
                    
                    edge_label = last_non_tag_action or "done"
                    self._add_edge(G, prev_node, label, edge_label)
                    prev_node = label
                    last_non_tag_action = None

                else:
                    # 非tag步骤
                    node_name = f"t{ti}_s{si}"
                    if node_name not in G:
                        G.add_node(node_name, kind="step", img_path=img_path,
                                  action_type=act_type, run_id=trace["run_id"])
                    
                    self._add_edge(G, prev_node, node_name, act_type)
                    prev_node = node_name
                    last_non_tag_action = act_type

        self.G = G
        
        # 计算一些统计信息
        tag_nodes = [n for n in G.nodes() if G.nodes[n].get('kind') == 'tag']
        step_nodes = [n for n in G.nodes() if G.nodes[n].get('kind') == 'step']
        
        print(f"[INFO] 图构建完成: {len(tag_nodes)}个tag节点, {len(step_nodes)}个step节点")
        print(f"[INFO] 总边数: {G.number_of_edges()}")
        
        return G

    # ---------- 4. 智能布局算法 ----------

    def compute_optimal_layout(self):
        """
        计算最优的自动布局
        使用层次布局 + 力导向优化
        """
        G = self.G
        if G.number_of_nodes() == 0:
            return {}
        
        # 1. 自动确定tag顺序
        tag_order = self._compute_tag_order_automatically()
        print(f"[INFO] 自动确定的tag顺序: {tag_order}")
        
        # 2. 为tag节点分配基础位置（水平线）
        tag_positions = {}
        for i, tag in enumerate(tag_order):
            if tag in G.nodes():
                x = i * self.layout_config['tag_spacing']
                y = 0  # 基础水平线
                tag_positions[tag] = np.array([x, y])
        
        # 3. 为step节点分配层次
        step_nodes = [n for n in G.nodes() if G.nodes[n].get('kind') == 'step']
        
        # 计算step节点到最近tag的距离
        tag_nodes = set(tag_positions.keys())
        step_levels = {}
        
        for step in step_nodes:
            # 找出连接到这个step的所有tag
            connected_tags = []
            for pred in G.predecessors(step):
                if pred in tag_nodes:
                    connected_tags.append(pred)
            for succ in G.successors(step):
                if succ in tag_nodes:
                    connected_tags.append(succ)
            
            if connected_tags:
                # 找到连接的tag在顺序中的平均位置
                tag_indices = [tag_order.index(tag) for tag in connected_tags 
                              if tag in tag_order]
                if tag_indices:
                    avg_index = np.mean(tag_indices)
                    # 根据连接关系确定层次
                    level = np.sign(avg_index - tag_order.index(connected_tags[0]))
                    step_levels[step] = level
                else:
                    step_levels[step] = 0
            else:
                step_levels[step] = 0
        
        # 4. 初始布局：tag在水平线，step在两侧
        initial_pos = tag_positions.copy()
        
        # 为step节点分配初始位置
        steps_by_level = defaultdict(list)
        for step, level in step_levels.items():
            steps_by_level[level].append(step)
        
        for level, steps in steps_by_level.items():
            base_x = np.mean([tag_positions[tag][0] for tag in tag_order 
                            if tag in tag_positions])
            base_y = level * self.layout_config['layer_gap']
            
            # 在水平方向上均匀分布
            for i, step in enumerate(steps):
                offset = (i - len(steps)/2) * self.layout_config['step_spacing']
                initial_pos[step] = np.array([base_x + offset, base_y])
        
        # 5. 使用力导向布局优化，固定tag节点
        try:
            # 使用更好的力导向参数
            pos = nx.spring_layout(
                G,
                pos=initial_pos,
                fixed=list(tag_positions.keys()),
                k=self.layout_config['tag_spacing'] / np.sqrt(G.number_of_nodes()),
                iterations=1000,
                seed=42,
                scale=2.0,  # 增加整体缩放
                center=[0, 0]
            )
            
            # 对step节点添加轻微抖动，增加可读性
            for node in G.nodes():
                if node not in tag_positions:
                    # 基于节点名的确定性抖动
                    seed = sum(ord(c) for c in str(node))
                    np.random.seed(seed % 1000)
                    jitter = np.random.uniform(
                        -self.layout_config['step_radius'],
                        self.layout_config['step_radius'],
                        2
                    )
                    pos[node] += jitter
            
        except Exception as e:
            print(f"[WARN] 力导向布局失败: {e}, 使用初始布局")
            pos = initial_pos
        
        # 6. 确保最小间距
        pos = self._enforce_minimum_distance(pos)
        
        self.pos = pos
        return pos

    def _enforce_minimum_distance(self, pos: Dict) -> Dict:
        """确保节点间的最小间距"""
        nodes = list(pos.keys())
        min_dist = self.layout_config['min_node_distance']
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n1, n2 = nodes[i], nodes[j]
                p1, p2 = pos[n1], pos[n2]
                dist = np.linalg.norm(p1 - p2)
                
                if dist < min_dist:
                    # 移动节点使其分开
                    direction = p2 - p1
                    if np.linalg.norm(direction) < 0.001:
                        direction = np.random.uniform(-1, 1, 2)
                    
                    direction = direction / np.linalg.norm(direction)
                    move = (min_dist - dist) / 2
                    
                    # 优先移动step节点
                    if self.G.nodes[n1].get('kind') == 'step':
                        pos[n1] -= direction * move
                    if self.G.nodes[n2].get('kind') == 'step':
                        pos[n2] += direction * move
        
        return pos

    # ---------- 5. UI框标注 ----------

    def annotate_clickable_boxes(self, pil_img: Image.Image, img_path: str) -> Image.Image:
        """在图片上标注UI可交互区域"""
        if not self.show_boxes:
            return pil_img
        
        if img_path in self._box_cache:
            bounds_list = self._box_cache[img_path]
        else:
            try:
                bounds_list = extract_all_bounds(img_path)
                self._box_cache[img_path] = bounds_list
            except Exception as e:
                print(f"[WARN] UI检测失败 {img_path}: {e}")
                bounds_list = []
        
        if not bounds_list:
            return pil_img
        
        # 创建半透明绿色框
        overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        for (left, top, right, bottom) in bounds_list:
            # 使用半透明的绿色
            draw.rectangle((left, top, right, bottom),
                          outline=(0, 255, 0, 180),  # 半透明绿色
                          width=2,
                          fill=(0, 255, 0, 30))     # 半透明填充
        
        # 合并原图和标注层
        pil_img = pil_img.convert('RGBA')
        result = Image.alpha_composite(pil_img, overlay)
        return result.convert('RGB')

    # ---------- 6. 美化绘图 ----------

    def draw_graph(self, show_edge_labels: bool = False, save_path: Optional[str] = None):
        """绘制优化后的美观图形 - tag节点也显示截图"""
        G = self.G
        pos = self.pos
        
        if not pos:
            print("[ERROR] 未计算布局")
            return
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(24, 16))
        fig.patch.set_facecolor('#f8f9fa')  # 浅灰色背景
        ax.set_facecolor('#ffffff')
        
        # 计算自适应参数
        num_nodes = G.number_of_nodes()
        base_node_size = max(8000 / (num_nodes ** 0.5), 200)
        
        # 定义美观的颜色方案
        color_palette = {
            'tag_bg': '#e3f2fd',      # 浅蓝色背景
            'tag_border': '#2196f3',   # 蓝色边框
            'step_bg': '#f3e5f5',      # 浅紫色背景
            'step_border': '#9c27b0',  # 紫色边框
            'start_color': '#4caf50',  # 绿色
            'done_color': '#f44336',   # 红色
            'start_highlight': '#a5d6a7',  # START节点高亮色
            'done_highlight': '#ef9a9a',   # done节点高亮色
            'tag_highlight': '#bbdefb',    # 普通tag高亮色
        }
        
        # 定义动作类型颜色
        action_colors = {
            'click': '#ff5252',    # 红色
            'swipe': '#4caf50',    # 绿色
            'input': '#ff9800',    # 橙色
            'wait': '#9e9e9e',     # 灰色
            'tag': '#2196f3',      # 蓝色
            'done': '#9c27b0',     # 紫色
            'unknown': '#607d8b',  # 蓝灰色
        }
        
        # --- 1. 先绘制边（在节点下面） ---
        edge_groups = defaultdict(list)
        for u, v, k, data in G.edges(keys=True, data=True):
            action_type = data.get('action', 'unknown')
            edge_groups[action_type].append((u, v, data))
        
        # 为每种动作类型绘制边
        for action_type, edges in edge_groups.items():
            color = action_colors.get(action_type, action_colors['unknown'])
            
            for u, v, data in edges:
                weight = data.get('weight', 1)
                linewidth = 1.0 + 0.8 * np.log1p(weight)
                alpha = 0.6 + 0.2 * min(weight / 5.0, 1.0)
                
                # 计算边弯曲度
                rad = 0.15
                if action_type == 'swipe':
                    rad = -0.15
                elif action_type == 'input':
                    rad = 0.25
                
                # 绘制边
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(u, v)],
                    edge_color=color,
                    arrows=True,
                    arrowsize=20,
                    arrowstyle='->',
                    width=linewidth,
                    alpha=alpha,
                    connectionstyle=f'arc3,rad={rad}',
                    ax=ax
                )
        
        # --- 2. 绘制节点 ---
        tag_nodes = [n for n in G.nodes() if G.nodes[n].get('kind') == 'tag']
        step_nodes = [n for n in G.nodes() if G.nodes[n].get('kind') == 'step']
        
        # 绘制tag节点（带截图）
        for n in tag_nodes:
            x, y = pos[n]
            img_path = G.nodes[n].get('img_path')
            
            # 确定节点类型和样式
            if n == "START":
                border_color = '#2e7d32'  # 深绿色边框
                glow_color = '#4caf50'    # 绿色光晕
                label_color = '#2e7d32'
            elif n == "done":
                border_color = '#c62828'  # 深红色边框
                glow_color = '#f44336'    # 红色光晕
                label_color = '#c62828'
            else:
                border_color = '#1976d2'  # 深蓝色边框
                glow_color = '#2196f3'    # 蓝色光晕
                label_color = '#1976d2'
            
            # 首先添加光晕效果
            for i in range(3, 0, -1):
                glow_radius = 1.5 + i * 0.2
                glow_alpha = 0.1 / i
                glow = plt.Circle((x, y), glow_radius,
                                facecolor=glow_color,
                                alpha=glow_alpha,
                                zorder=1 + i)
                ax.add_patch(glow)
            
            # 添加背景装饰圆（外圈）
            decoration = plt.Circle((x, y), 1.5,
                                facecolor='none',
                                edgecolor=border_color,
                                linewidth=3,
                                alpha=0.3,
                                zorder=2)
            ax.add_patch(decoration)
            
            # 如果tag节点有截图，显示截图
            if img_path and os.path.isfile(img_path):
                try:
                    # 加载并处理图片
                    img = Image.open(img_path).convert('RGB')
                    
                    # tag节点的图片比step节点大一些
                    img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                    
                    # 如果需要，添加UI框
                    if self.show_boxes:
                        img = self.annotate_clickable_boxes(img, img_path)
                    
                    # 创建圆角效果
                    img_array = np.asarray(img)
                    
                    # 添加一个圆形遮罩让图片变成圆形
                    h, w = img_array.shape[:2]
                    mask = Image.new('L', (w, h), 0)
                    draw = ImageDraw.Draw(mask)
                    draw.ellipse((0, 0, w, h), fill=255)
                    
                    # 将图片转换为RGBA并应用圆形遮罩
                    img_rgba = Image.new('RGBA', (w, h), (255, 255, 255, 0))
                    img_rgba.paste(img.convert('RGBA'), (0, 0), mask)
                    
                    # 创建OffsetImage
                    im = OffsetImage(np.asarray(img_rgba), zoom=0.2)
                    
                    # 添加带装饰的边框
                    ab = AnnotationBbox(
                        im,
                        (x, y),
                        frameon=True,
                        pad=0.08,  # 比step节点更大的padding
                        bboxprops=dict(
                            boxstyle='round,pad=0.08',
                            edgecolor=border_color,
                            linewidth=3,  # 更粗的边框
                            facecolor='white',
                            alpha=1.0
                        )
                    )
                    ax.add_artist(ab)
                    
                    # 在图片下方添加标签背景
                    label_bg = plt.Rectangle((x-0.8, y-2.0), 1.6, 0.6,
                                        facecolor='white',
                                        edgecolor='none',
                                        alpha=0.9,
                                        zorder=4)
                    ax.add_patch(label_bg)
                    
                    # 添加tag标签文字（在图片下方）
                    ax.text(x, y-1.7, n,
                        fontsize=11,
                        fontweight='bold',
                        ha='center',
                        va='center',
                        color=label_color,
                        zorder=5)
                    
                except Exception as e:
                    print(f"[WARN] 无法加载tag节点图片 {img_path}: {e}")
                    # 回退方案：绘制装饰性圆形
                    self._draw_decorative_tag_circle(ax, x, y, n, border_color, glow_color)
            else:
                # 无图片时的回退：绘制装饰性圆形
                self._draw_decorative_tag_circle(ax, x, y, n, border_color, glow_color)
        
        # 绘制step节点（带截图）
        for n in step_nodes:
            x, y = pos[n]
            img_path = G.nodes[n].get('img_path')
            action_type = G.nodes[n].get('action_type', 'unknown')
            
            # 根据动作类型确定边框颜色
            border_color = action_colors.get(action_type, action_colors['unknown'])
            
            if img_path and os.path.isfile(img_path):
                try:
                    # 加载并处理图片
                    img = Image.open(img_path).convert('RGB')
                    
                    # 调整图片大小（比tag节点小）
                    img.thumbnail((160, 160), Image.Resampling.LANCZOS)
                    
                    # 如果需要，添加UI框
                    if self.show_boxes:
                        img = self.annotate_clickable_boxes(img, img_path)
                    
                    # 创建圆角效果
                    img_array = np.asarray(img)
                    
                    # 添加一个圆角矩形遮罩
                    h, w = img_array.shape[:2]
                    mask = Image.new('L', (w, h), 0)
                    draw = ImageDraw.Draw(mask)
                    # 绘制圆角矩形
                    radius = min(w, h) // 4
                    draw.rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
                    
                    # 将图片转换为RGBA并应用圆角遮罩
                    img_rgba = Image.new('RGBA', (w, h), (255, 255, 255, 0))
                    img_rgba.paste(img.convert('RGBA'), (0, 0), mask)
                    
                    # 创建OffsetImage
                    im = OffsetImage(np.asarray(img_rgba), zoom=0.16)
                    
                    # 添加边框
                    ab = AnnotationBbox(
                        im,
                        (x, y),
                        frameon=True,
                        pad=0.05,
                        bboxprops=dict(
                            boxstyle='round,pad=0.05',
                            edgecolor=border_color,
                            linewidth=2,
                            facecolor='white',
                            alpha=0.95
                        )
                    )
                    ax.add_artist(ab)
                    
                    # 添加动作类型的小标签
                    if action_type and action_type != 'unknown':
                        # 小背景
                        action_bg = plt.Rectangle((x-0.5, y-1.0), 1.0, 0.4,
                                            facecolor=border_color,
                                            edgecolor='none',
                                            alpha=0.8,
                                            zorder=4)
                        ax.add_patch(action_bg)
                        
                        # 动作类型文字
                        ax.text(x, y-0.8, action_type[:4],
                            fontsize=7,
                            fontweight='bold',
                            ha='center',
                            va='center',
                            color='white',
                            zorder=5)
                    
                except Exception as e:
                    print(f"[WARN] 无法加载step节点图片 {img_path}: {e}")
                    # 回退方案：绘制简单圆形
                    circle = plt.Circle((x, y), 0.7,
                                    facecolor='#f3e5f5',
                                    edgecolor=border_color,
                                    linewidth=2,
                                    alpha=0.8)
                    ax.add_patch(circle)
                    ax.text(x, y, action_type[:3],
                        fontsize=8,
                        ha='center',
                        va='center')
            else:
                # 无图片时的回退
                circle = plt.Circle((x, y), 0.7,
                                facecolor='#f3e5f5',
                                edgecolor=border_color,
                                linewidth=2,
                                alpha=0.8)
                ax.add_patch(circle)
                ax.text(x, y, action_type[:3] if action_type != 'unknown' else '?',
                    fontsize=8,
                    ha='center',
                    va='center')
        
        # --- 3. 添加边标签（可选） ---
        if show_edge_labels:
            edge_labels = {}
            for u, v, k, data in G.edges(keys=True, data=True):
                action_type = data.get('action', 'unknown')
                if action_type == 'click':
                    continue  # 跳过click标签，太密集
                
                weight = data.get('weight', 1)
                if weight > 1:
                    text = f"{action_type}×{weight}"
                else:
                    text = action_type
                
                edge_labels[(u, v)] = text
            
            if edge_labels:
                nx.draw_networkx_edge_labels(
                    G, pos,
                    edge_labels=edge_labels,
                    font_size=8,
                    font_weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2',
                            facecolor='white',
                            alpha=0.8,
                            edgecolor='none'),
                    ax=ax
                )
        
        # --- 4. 添加图例 ---
        legend_elements = []
        for action_type, color in action_colors.items():
            if action_type in ['tag', 'done']:
                continue
            legend_elements.append(
                plt.Line2D([0], [0], 
                        color=color, 
                        lw=3,
                        label=f'{action_type}',
                        alpha=0.8)
            )
        
        if legend_elements:
            legend = ax.legend(
                handles=legend_elements,
                loc='upper left',
                bbox_to_anchor=(1.02, 1.0),
                title='动作类型',
                fontsize=10,
                title_fontsize=11,
                frameon=True,
                fancybox=True,
                shadow=True,
                facecolor='white',
                edgecolor='lightgray'
            )
            legend.get_frame().set_alpha(0.9)
        
        # --- 5. 添加统计信息 ---
        stats_text = (f"应用: {self.app} | 任务: {self.task}\n"
                    f"轨迹数: {len(self.traces)}\n"
                    f"总节点: {G.number_of_nodes()} | "
                    f"总边数: {G.number_of_edges()}")
        
        ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5',
                        facecolor='white',
                        alpha=0.8,
                        edgecolor='lightgray'))
        
        # --- 6. 美化坐标轴 ---
        ax.set_xlim(min(x for x, _ in pos.values()) - 6,
                max(x for x, _ in pos.values()) + 6)
        ax.set_ylim(min(y for _, y in pos.values()) - 6,
                max(y for _, y in pos.values()) + 6)
        
        ax.set_title(f'{self.app} - {self.task} 交互状态机',
                    fontsize=16,
                    fontweight='bold',
                    pad=20)
        
        # 隐藏坐标轴
        ax.axis('off')
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                    facecolor=fig.get_facecolor())
            print(f"[INFO] 图形已保存到: {save_path}")
        else:
            plt.show()
        
        return fig

    def _draw_decorative_tag_circle(self, ax, x, y, label, border_color, glow_color):
        """
        绘制装饰性的tag节点圆形（无截图时的回退方案）
        """
        # 添加光晕效果
        for i in range(3, 0, -1):
            glow_radius = 1.5 + i * 0.2
            glow_alpha = 0.1 / i
            glow = plt.Circle((x, y), glow_radius,
                            facecolor=glow_color,
                            alpha=glow_alpha,
                            zorder=1 + i)
            ax.add_patch(glow)
        
        # 添加渐变背景圆形
        gradient_circle = plt.Circle((x, y), 1.2,
                                facecolor=glow_color,
                                alpha=0.3,
                                zorder=2)
        ax.add_patch(gradient_circle)
        
        # 添加主圆形
        main_circle = plt.Circle((x, y), 1.0,
                            facecolor='white',
                            edgecolor=border_color,
                            linewidth=3,
                            alpha=0.9,
                            zorder=3)
        ax.add_patch(main_circle)
        
        # 添加内圈装饰
        inner_circle = plt.Circle((x, y), 0.7,
                                facecolor=glow_color,
                                alpha=0.2,
                                zorder=4)
        ax.add_patch(inner_circle)
        
        # 添加标签文字
        ax.text(x, y, label,
            fontsize=12,
            fontweight='bold',
            ha='center',
            va='center',
            color=border_color,
            zorder=5)
        
        # 添加标签下方的小文字（如果是START或done）
        if label == "START":
            subtext = "开始"
        elif label == "done":
            subtext = "结束"
        else:
            subtext = "状态"
        
        ax.text(x, y-1.4, subtext,
            fontsize=9,
            ha='center',
            va='top',
            color=border_color,
            zorder=4,
            bbox=dict(boxstyle='round,pad=0.2',
                        facecolor='white',
                        alpha=0.8,
                        edgecolor='none'))


        # ---------- 7. 主运行方法 ----------

    def run(self, show_edge_labels: bool = False, save_path: Optional[str] = None):
        """主运行方法"""
        print(f"[INFO] 开始可视化: {self.app}/{self.task}")
        
        # 加载数据
        if not self.load_runs():
            print("[ERROR] 未找到有效轨迹数据")
            return
        
        # 构建图
        self.build_graph()
        
        # 自动计算布局
        print("[INFO] 正在计算自动布局...")
        self.compute_optimal_layout()
        
        # 绘制图形
        print("[INFO] 正在生成可视化...")
        self.draw_graph(
            show_edge_labels=show_edge_labels,
            save_path=save_path
        )
        
        print("[INFO] 可视化完成！")


# ---------- CLI接口 ----------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="自动布局的FSM可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python multi_vis_tagstep_with_boxes_auto.py \\
    --data_path ./data \\
    --app 携程 \\
    --task hotel_booking \\
    --show-boxes \\
    --edge-labels \\
    --save ./output/fsm_auto.png
        """
    )
    
    parser.add_argument(
        "--data_path",
        required=True,
        help="数据根目录路径"
    )
    
    parser.add_argument(
        "--app",
        required=True,
        help="应用名称"
    )
    
    parser.add_argument(
        "--task",
        required=True,
        help="任务名称"
    )
    
    parser.add_argument(
        "--edge-labels",
        action="store_true",
        help="显示边标签（除了click类型）"
    )
    
    parser.add_argument(
        "--show-boxes",
        action="store_true",
        help="在截图上显示UI检测框"
    )
    
    parser.add_argument(
        "--save",
        default=None,
        help="保存图片的路径（可选）"
    )
    
    args = parser.parse_args()
    
    # 创建可视化器并运行
    visualizer = AutoLayoutFSMVisualizer(
        data_root=args.data_path,
        app=args.app,
        task=args.task,
        show_boxes=args.show_boxes
    )
    
    visualizer.run(
        show_edge_labels=args.edge_labels,
        save_path=args.save
    )