# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Plotting functions."""

import colorlover as cl
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as offline
import xnas.core.logging as logging
from graphviz import Digraph


# def get_plot_colors(max_colors, color_format="pyplot"):
#     """Generate colors for plotting."""
#     colors = cl.scales["11"]["qual"]["Paired"]
#     if max_colors > len(colors):
#         colors = cl.to_rgb(cl.interp(colors, max_colors))
#     if color_format == "pyplot":
#         return [[j / 255.0 for j in c] for c in cl.to_numeric(colors)]
#     return colors


# def prepare_plot_data(log_files, names, metric="top1_err"):
#     """Load logs and extract data for plotting error curves."""
#     plot_data = []
#     for file, name in zip(log_files, names):
#         d, data = {}, logging.sort_log_data(logging.load_log_data(file))
#         for phase in ["train", "test"]:
#             x = data[phase + "_epoch"]["epoch"]
#             y = data[phase + "_epoch"][metric]
#             x = [int(e.split("/")[0]) for e in x]
#             d["x_" + phase], d["y_" + phase] = x, y
#             d[phase + "_label"] = "[{:5.2f}] ".format(min(y) if y else 0) + name
#         plot_data.append(d)
#     assert len(plot_data) > 0, "No data to plot"
#     return plot_data


# def plot_error_curves_plotly(log_files, names, filename, metric="top1_err"):
#     """Plot error curves using plotly and save to file."""
#     plot_data = prepare_plot_data(log_files, names, metric)
#     colors = get_plot_colors(len(plot_data), "plotly")
#     # Prepare data for plots (3 sets, train duplicated w and w/o legend)
#     data = []
#     for i, d in enumerate(plot_data):
#         s = str(i)
#         line_train = {"color": colors[i], "dash": "dashdot", "width": 1.5}
#         line_test = {"color": colors[i], "dash": "solid", "width": 1.5}
#         data.append(
#             go.Scatter(
#                 x=d["x_train"],
#                 y=d["y_train"],
#                 mode="lines",
#                 name=d["train_label"],
#                 line=line_train,
#                 legendgroup=s,
#                 visible=True,
#                 showlegend=False,
#             )
#         )
#         data.append(
#             go.Scatter(
#                 x=d["x_test"],
#                 y=d["y_test"],
#                 mode="lines",
#                 name=d["test_label"],
#                 line=line_test,
#                 legendgroup=s,
#                 visible=True,
#                 showlegend=True,
#             )
#         )
#         data.append(
#             go.Scatter(
#                 x=d["x_train"],
#                 y=d["y_train"],
#                 mode="lines",
#                 name=d["train_label"],
#                 line=line_train,
#                 legendgroup=s,
#                 visible=False,
#                 showlegend=True,
#             )
#         )
#     # Prepare layout w ability to toggle 'all', 'train', 'test'
#     titlefont = {"size": 18, "color": "#7f7f7f"}
#     vis = [[True, True, False], [False, False, True], [False, True, False]]
#     buttons = zip(["all", "train", "test"], [[{"visible": v}] for v in vis])
#     buttons = [{"label": b, "args": v, "method": "update"} for b, v in buttons]
#     layout = go.Layout(
#         title=metric + " vs. epoch<br>[dash=train, solid=test]",
#         xaxis={"title": "epoch", "titlefont": titlefont},
#         yaxis={"title": metric, "titlefont": titlefont},
#         showlegend=True,
#         hoverlabel={"namelength": -1},
#         updatemenus=[
#             {
#                 "buttons": buttons,
#                 "direction": "down",
#                 "showactive": True,
#                 "x": 1.02,
#                 "xanchor": "left",
#                 "y": 1.08,
#                 "yanchor": "top",
#             }
#         ],
#     )
#     # Create plotly plot
#     offline.plot({"data": data, "layout": layout}, filename=filename)


# def plot_error_curves_pyplot(log_files, names, filename=None, metric="top1_err"):
#     """Plot error curves using matplotlib.pyplot and save to file."""
#     plot_data = prepare_plot_data(log_files, names, metric)
#     colors = get_plot_colors(len(names))
#     for ind, d in enumerate(plot_data):
#         c, lbl = colors[ind], d["test_label"]
#         plt.plot(d["x_train"], d["y_train"], "--", c=c, alpha=0.8)
#         plt.plot(d["x_test"], d["y_test"], "-", c=c, alpha=0.8, label=lbl)
#     plt.title(metric + " vs. epoch\n[dash=train, solid=test]", fontsize=14)
#     plt.xlabel("epoch", fontsize=14)
#     plt.ylabel(metric, fontsize=14)
#     plt.grid(alpha=0.4)
#     plt.legend()
#     if filename:
#         plt.savefig(filename)
#         plt.clf()
#     else:
#         plt.show()


def graph_plot(genotype, file_path, caption=None, search_space='darts', n_nodes=None):
    """ make DAG plot and save to file_path as .png """
    edge_attr = {
        'fontsize': '20',
        'fontname': 'times'
    }
    node_attr = {
        'style': 'filled',
        'shape': 'rect',
        'align': 'center',
        'fontsize': '20',
        'height': '0.5',
        'width': '0.5',
        'penwidth': '2',
        'fontname': 'times'
    }
    g = Digraph(
        format='png',
        edge_attr=edge_attr,
        node_attr=node_attr,
        engine='dot')
    g.body.extend(['rankdir=LR'])

    if search_space in ['darts', 'nasbench301']:
        # input nodes
        g.node("c_{k-2}", fillcolor='darkseagreen2')
        g.node("c_{k-1}", fillcolor='darkseagreen2')

        # intermediate nodes
        n_nodes = len(genotype)
        for i in range(n_nodes):
            g.node(str(i), fillcolor='lightblue')

        for i, edges in enumerate(genotype):
            for op, j in edges:
                if j == 0:
                    u = "c_{k-2}"
                elif j == 1:
                    u = "c_{k-1}"
                else:
                    u = str(j-2)

                v = str(i)
                g.edge(u, v, label=op, fillcolor="gray")

        # output node
        g.node("c_{k}", fillcolor='palegoldenrod')
        for i in range(n_nodes):
            g.edge(str(i), "c_{k}", fillcolor="gray")
    elif search_space == 'nasbench_201':
        # input node
        g.node("0", fillcolor='darkseagreen2')
        if n_nodes is None:
            n_nodes = 3
        for i in range(n_nodes):
            g.node(str(i+1), fillcolor='lightblue')
        genotype = genotype.split('+')
        for i, node_str in enumerate(genotype):
            edges = node_str.split('|')
            edges = edges[1:-1]
            for edge in edges:
                op = edge[0:-2]
                u = edge[-1]
                v = str(i+1)
                g.edge(u, v, label=op, fillcolor="gray")
    else:
        raise NotImplementedError

    # add image caption
    if caption:
        g.attr(label=caption, overlap='false', fontsize='20', fontname='times')

    g.render(file_path, view=False)
