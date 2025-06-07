import plotly.graph_objects as go
import copy
import numpy as np
from pathlib import Path
import plotly.express as px
from .utils import apply_opacity_to_rgb


def plot_graph(
        X=None,
        labels=None,
        edges=None,
        cluster_centers=None,
        label_groups=None,
        fig_title=None,
        preds=None,
        x_grid=None,
        y_grid=None,
        fine_grained=False,
        fine_grained_x=False,
        fine_grained_center=False,
        train_indices=None,
        layout_kwargs=None,
        dump_dir=None,
        display=False,
        node_size=10
):
    """
    绘制节点及其之间的边。添加了绘制簇心的功能。

    参数:
    - X: 节点位置的二维数组，形状为 (n_nodes, 2)
    - labels: 节点的类标签
    - edges: 边列表，每个边为 (i, j, edge_type)，其中 i 和 j 为节点索引，edge_type 为边类型（'intra' 或 'inter'）
    - draw_edges: 是否绘制边
    - cluster_centers: 簇心位置的二维数组，形状为 (n_clusters, 2)（可选）
    """
    node_indices = np.arange(len(X))
    node_mask = np.ones(len(node_indices), dtype=bool)
    train_indices = np.asarray(train_indices) if train_indices is not None else None
    train_mask = None
    if train_indices is not None:
        if np.issubsctype(train_indices.dtype, np.bool_):
            train_mask = copy.deepcopy(train_indices)
            train_indices = np.nonzero(train_mask)[0]
        else:
            train_mask = np.zeros(len(node_indices), dtype=bool)
            train_mask[train_indices] = True

    layout_kwargs = layout_kwargs or {}
    assert X.shape[1] == 2
    # 基于节点属性和标签绘制散点图
    unique_labels = np.unique(labels)

    discrete_colors = px.colors.qualitative.Set1  # Use the 'Plotly' discrete color scale
    # Assign colors based on the unique labels
    label_to_color = {label: discrete_colors[i % len(discrete_colors)] for i, label in enumerate(unique_labels)}

    fig = go.Figure()
    # 决策边界
    dbo = decision_boundary_opacity = 0.65
    contour_discrete_colorscale = [apply_opacity_to_rgb(color, dbo) for color in discrete_colors]

    if preds is not None:
        fig.add_trace(
            go.Contour(
                x=x_grid,
                y=y_grid,
                z=preds,
                # opacity=dbo,
                showscale=True,
                name="Decision Boundary",
                contours=dict(start=0, end=len(unique_labels) - 1, size=1, coloring="fill"),
                colorscale=contour_discrete_colorscale,
                colorbar=dict(title="Decision Region", orientation="h"),
            )
        )

    # ========================================= 数据点
    dp_shapes = ['circle', 'square', 'diamond', 'triangle-up']
    if X is not None:
        if train_mask is not None:
            dp_symbols = np.asarray(dp_shapes)[train_mask.astype(int)]
        else:
            dp_symbols = dp_shapes[0]  # circle
        if not (fine_grained or fine_grained_x):
            # ======================================== style 1
            node_colors = [label_to_color[lbl] for lbl in labels]
            t1 = go.Scatter(
                x=X[:, 0],
                y=X[:, 1],
                mode="markers",
                marker=dict(
                    symbol=dp_symbols,
                    size=node_size,
                    color=node_colors,  # Use labels as colors
                    line=dict(color="black", width=1),
                ),
                name="Nodes",
                text=[f"Input: ({X[i, 0]:.4f},{X[i, 1]:.4f}) | Class: {label}" for i, label in enumerate(labels)],
                # Hover text
                hoverinfo="text",
                # visible="legendonly",
            )
            fig.add_trace(t1)
        # ========================================= 数据点
        else:
            # ======================================== style 2
            # Add scatter plot traces for each label
            for label in unique_labels:
                indices = np.where(labels == label)
                X1 = X[indices, 0][0]
                X2 = X[indices, 1][0]
                if train_mask is not None:
                    mask_sub = train_mask[indices]
                    dp_symbols = np.asarray(dp_shapes)[mask_sub.astype(int)]
                else:
                    dp_symbols = dp_shapes[0]  # circle
                fig.add_trace(
                    go.Scatter(
                        x=X1,
                        y=X2,
                        mode="markers",
                        marker=dict(
                            symbol=dp_symbols,
                            size=node_size,
                            color=label_to_color[label],  # Use discrete colors from the palette
                            line=dict(color="black", width=1),
                        ),
                        name=f"Label {label}",  # Legend label
                        text=[f"Input: ({X1[i]:.4f},{X2[i]:.4f}) | Class: {label}" for i in range(len(indices[0]))],
                        # Hover text
                        hoverinfo="text",
                        # legendgroup='Nodes',
                        showlegend=True,
                    )
                )

    if edges is not None:
        edge_x_intra = []
        edge_y_intra = []
        edge_x_inter = []
        edge_y_inter = []

        for edge in edges:
            i, j, edge_type = edge
            if edge_type == "intra":
                edge_x_intra += [X[i, 0], X[j, 0], None]
                edge_y_intra += [X[i, 1], X[j, 1], None]
            elif edge_type == "inter":
                edge_x_inter += [X[i, 0], X[j, 0], None]
                edge_y_inter += [X[i, 1], X[j, 1], None]

        fig.add_trace(
            go.Scatter(
                x=edge_x_intra,
                y=edge_y_intra,
                mode="lines",
                line=dict(color="blue", width=1),
                opacity=0.5,
                name="Intra-class Edges",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=edge_x_inter,
                y=edge_y_inter,
                mode="lines",
                line=dict(color="gray", width=1),
                opacity=0.4,
                name="Inter-class Edges",
            )
        )

    # 添加簇心到图中
    if cluster_centers is not None:
        if not (fine_grained or fine_grained_center):
            symbols = []
            texts = []
            for cc in unique_labels:
                if label_groups is not None:
                    lbl_group = label_groups[cc]
                else:
                    lbl_group = 0
                syb = "x" if lbl_group == 0 else "star"
                symbols.append(syb)
                texts.append(f"Center {cc}")

            center_colors = [label_to_color[lbl] for lbl in unique_labels]
            center_scatter = go.Scatter(
                x=cluster_centers[:, 0],
                y=cluster_centers[:, 1],
                mode="markers",
                marker=dict(
                    size=node_size * 2.5,
                    color=center_colors,
                    symbol=symbols,
                    line=dict(width=2),
                ),
                name=f"Centers",
                text=texts,
                hoverinfo="text",
            )
            fig.add_trace(center_scatter)
        else:
            for i, center in enumerate(cluster_centers):
                # 获取对应簇心的类标签
                lbl = unique_labels[i] if i < len(unique_labels) else None
                if label_groups is not None:
                    lbl_group = label_groups[lbl]
                else:
                    lbl_group = 0
                # 计算簇心的颜色，与对应类别的颜色一致
                symbol = "x" if lbl_group == 0 else "star"
                fig.add_trace(
                    go.Scatter(
                        x=[center[0]],
                        y=[center[1]],
                        mode="markers",
                        marker=dict(
                            size=node_size * 2.5,
                            color=label_to_color[lbl],
                            symbol=symbol,
                            line=dict(width=2),
                        ),
                        name=f"Center {lbl}",
                        text=f"Center {lbl}",
                        hoverinfo="text",
                    )
                )

    # 设置X轴和Y轴尺度一致
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.update_layout(
        title=f"{fig_title}",
        showlegend=True,
        **layout_kwargs,
    )

    if x_grid is not None and y_grid is not None:
        fig.update_xaxes(range=[min(x_grid), max(x_grid)])
        fig.update_yaxes(range=[min(y_grid), max(y_grid)])

    if dump_dir is not None:
        dump_dir = Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        fig.write_html(dump_dir / f"{fig_title}.html")

    if display:
        fig.show()
