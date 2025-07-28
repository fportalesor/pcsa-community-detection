import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import mapclassify
import numpy as np
import networkx as nx
from shapely.geometry import Point
import contextily as ctx

class GraphPlotter:

    def __init__(self):
        self.fig = None
        self.ax = None

    def plot_graph(
        self,
        graph,
        bbox,
        polygon_gdf=None,
        weights_col="weight",
        vertex_color="skyblue",
        cmap_name="OrRd",
        k_classes=5,
        figsize=(8, 8),
        add_basemap=False,
        basemap_provider=ctx.providers.Esri.WorldImagery,
        basemap_attribution="Tiles (C) Esri",
        basemap_attribution_size=6,
        node_size=1,
        edge_width_min=0.5,
        edge_width_max=3,
        edge_alpha_min=0.1,
        edge_alpha_max=1.0,
        polygon_edgecolor="lightgrey",
        polygon_linewidth=0.5,
        legend_title="Edge weights (Jenks)",
        zoom_to_nodes=True,
        filter_nodes=None,
        title=None
    ):
        coord_map = {
            str(n): (data["lon"], data["lat"])
            for n, data in graph.nodes(data=True)
        }

        graph = self._remove_self_loops(graph)
        filtered_vertices = self._filter_vertices_in_polygon(graph, coord_map, bbox)

        if filter_nodes is not None:
            filter_nodes_set = set(str(n) for n in filter_nodes)
            filtered_vertices = [n for n in filtered_vertices if n in filter_nodes_set]

        subgraph = graph.subgraph(filtered_vertices).copy()
        filtered_edges = self._filter_edges_in_polygon(subgraph, coord_map, bbox)
        subgraph = subgraph.edge_subgraph(filtered_edges).copy()

        layout = np.array([coord_map[n] for n in subgraph.nodes()])
        edge_styles = self._style_edges(
            subgraph, weights_col, cmap_name, k_classes,
            edge_width_min, edge_width_max,
            edge_alpha_min, edge_alpha_max
        )

        self._plot(
            subgraph, layout, polygon_gdf, edge_styles,
            vertex_color, figsize, edge_styles["legend"],
            add_basemap, basemap_provider, basemap_attribution, basemap_attribution_size,
            node_size, polygon_edgecolor, polygon_linewidth, legend_title,
            zoom_to_nodes, title
        )

    def _remove_self_loops(self, graph):
        graph_no_self = graph.copy()
        graph_no_self.remove_edges_from(nx.selfloop_edges(graph_no_self))
        return graph_no_self

    def _filter_vertices_in_polygon(self, graph, coord_map, geometry):
        return [
            n for n in graph.nodes()
            if n in coord_map and geometry.contains(Point(coord_map[n]))
        ]

    def _filter_edges_in_polygon(self, graph, coord_map, geometry):
        return [
            (u, v) for u, v in graph.edges()
            if geometry.contains(Point(coord_map[u])) and geometry.contains(Point(coord_map[v]))
        ]

    def _style_edges(self, graph, weights_col, cmap_name, k_classes,
                     edge_width_min, edge_width_max, edge_alpha_min, edge_alpha_max):
        weights = [graph[u][v].get(weights_col, 1) for u, v in graph.edges()]
        classifier = mapclassify.NaturalBreaks(weights, k=k_classes)
        classes = classifier.yb
        cmap = cm.get_cmap(cmap_name, k_classes)

        edge_colors = [np.clip(cmap(cls), 0, 1) for cls in classes]

        edge_widths = [
            edge_width_min + ((cls / (k_classes - 1)) ** 4) * (edge_width_max - edge_width_min)
            for cls in classes
        ]
        edge_alphas = [
            edge_alpha_min + ((cls / (k_classes - 1)) ** 2) * (edge_alpha_max - edge_alpha_min)
            for cls in classes
        ]
        edge_colors_with_alpha = [
            (r, g, b, min(max(alpha, 0), 1))
            for (r, g, b, _), alpha in zip(edge_colors, edge_alphas)
        ]

        curved = [(-1) ** i * 0.2 for i in range(len(graph.edges()))]

        # Create a list of edge drawing instructions
        styled_edges = []
        for i, ((u, v), color, width, alpha, curve, cls) in enumerate(zip(
            graph.edges(), edge_colors_with_alpha, edge_widths, edge_alphas, curved, classes
        )):
            styled_edges.append({
                "edge": (u, v),
                "color": color,
                "width": width,
                "alpha": alpha,
                "curve": curve,
                "class": cls
            })

        styled_edges_sorted = sorted(styled_edges, key=lambda x: x["class"])

        legend_labels = [
            f"{int(min(weights)) if i == 0 else int(classifier.bins[i-1])}-{int(classifier.bins[i])}"
            for i in range(k_classes)
        ]
        legend_colors = [cmap(i)[:3] + (1.0,) for i in range(k_classes)]
        legend_patches = [
            mpatches.Patch(color=color, label=label)
            for color, label in zip(legend_colors, legend_labels)
        ]

        return {
            "edges": styled_edges_sorted,
            "legend": legend_patches
        }

    def _plot(self, graph, layout, polygon_gdf, edge_styles, vertex_color, figsize, legend_patches,
              add_basemap=False, basemap_provider=None, basemap_attribution="", basemap_attribution_size=6,
              node_size=1, polygon_edgecolor="lightgrey", polygon_linewidth=0.5,
              legend_title="Edge weights (Jenks)", zoom_to_nodes=False, title=None):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        ax = self.ax

        if title:
            self.ax.set_title(title, fontsize=12)

        if polygon_gdf is not None:
            polygon_gdf.boundary.plot(ax=ax, edgecolor=polygon_edgecolor,
                                       linewidth=polygon_linewidth, zorder=1)

        for edge_data in edge_styles["edges"]:
            u, v = edge_data["edge"]
            x0, y0 = layout[list(graph.nodes()).index(u)]
            x1, y1 = layout[list(graph.nodes()).index(v)]

            dx, dy = x1 - x0, y1 - y0
            ctrl_x = (x0 + x1) / 2 + edge_data["curve"] * dy
            ctrl_y = (y0 + y1) / 2 - edge_data["curve"] * dx

            path = mpath.Path(
                [(x0, y0), (ctrl_x, ctrl_y), (x1, y1)],
                [mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3]
            )
            patch = mpatches.PathPatch(
                path,
                edgecolor=edge_data["color"],
                linewidth=edge_data["width"],
                alpha=edge_data["alpha"],
                zorder=2,
                facecolor='none'
            )
            ax.add_patch(patch)

        xs, ys = layout[:, 0], layout[:, 1]
        ax.scatter(xs, ys, s=node_size, color=vertex_color, zorder=3)

        if zoom_to_nodes:
            margin = 100
            ax.set_xlim(xs.min() - margin, xs.max() + margin)
            ax.set_ylim(ys.min() - margin, ys.max() + margin)

        ax.legend(
            handles=legend_patches,
            title=legend_title,
            loc="upper right",
            fontsize="small",
            title_fontsize="medium",
            frameon=True
        )

        if add_basemap and basemap_provider is not None:
            try:
                ctx.add_basemap(
                    ax,
                    source=basemap_provider,
                    crs="EPSG:32719",
                    attribution=basemap_attribution,
                    attribution_size=basemap_attribution_size
                )
            except Exception as e:
                print(f"Could not add basemap: {e}")

        ax.set_axis_off()
        plt.tight_layout()

    def save_plot(self, path, dpi=300, bbox_inches="tight"):
        if self.fig:
            self.fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        else:
            raise RuntimeError("No figure to save. Run plot_graph() first.")