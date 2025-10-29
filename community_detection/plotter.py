import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import mapclassify
import numpy as np
import networkx as nx
from shapely.geometry import Point
import contextily as ctx
from matplotlib.colors import to_rgba

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
    show_edge_legend=True,
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
    polygon_category=None,
    polygon_cmap="tab20",
    polygon_alpha=0.5,
    legend_title="Edge weights (Jenks)",
    zoom_to_nodes=True,
    filter_nodes=None,
    title=None,
    health_centre_df=None,
    health_centre_id_col="TractID",
    health_centre_value_col="total_visits",
    health_node_k_classes=5,
    health_node_color="dodgerblue",
    health_node_size_min=20,
    health_node_size_max=100,
    health_node_alpha=1.0
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

    if health_centre_df is not None:
      health_ids = set(health_centre_df[health_centre_id_col].astype(str))
      for node_id in filtered_vertices:
        if node_id in health_ids and node_id not in subgraph:
          subgraph.add_node(node_id, **graph.nodes[node_id])

    layout = np.array([coord_map[n] for n in subgraph.nodes()])
    edge_styles = self._style_edges(
      subgraph, weights_col, cmap_name, k_classes,
      edge_width_min, edge_width_max, edge_alpha_min, edge_alpha_max
    )

    legend_patches = edge_styles["legend"] if show_edge_legend else []

    self._plot(
      subgraph, layout, polygon_gdf, edge_styles,
      vertex_color, figsize, legend_patches,
      add_basemap, basemap_provider, basemap_attribution, basemap_attribution_size,
      node_size, polygon_edgecolor, polygon_linewidth, polygon_category,
      polygon_cmap, polygon_alpha, legend_title,
      zoom_to_nodes, title,
      health_centre_df, health_centre_id_col, health_centre_value_col,
      health_node_k_classes, health_node_color,
      health_node_size_min, health_node_size_max, health_node_alpha
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

    edge_data = []
    for (u, v), cls in zip(graph.edges(), classes):
      width = edge_width_min + ((cls / (k_classes - 1)) ** 4) * (edge_width_max - edge_width_min)
      alpha = edge_alpha_min + ((cls / (k_classes - 1)) ** 2) * (edge_alpha_max - edge_alpha_min)
      curve = (-1) ** (len(edge_data)) * 0.2
      color = cmap(cls)
      edge_data.append({
        "edge": (u, v),
        "color": color,
        "width": width,
        "alpha": alpha,
        "curve": curve,
        "class": cls
      })

    edge_data.sort(key=lambda x: x["class"])  # Ensure correct draw order

    def format_bin(value):
      return f"{int(value)}" if float(value).is_integer() else f"{value:.2f}"
    
    legend_labels = [
      f"{format_bin(min(weights))}-{format_bin(classifier.bins[i])}" if i == 0
      else f"{format_bin(classifier.bins[i - 1])}-{format_bin(classifier.bins[i])}"
      for i in range(k_classes)
    ]

    legend_colors = [cmap(i)[:3] + (1.0,) for i in range(k_classes)]
    legend_patches = [
      mpatches.Patch(color=color, label=label)
      for color, label in zip(legend_colors, legend_labels)
    ]

    return {"edges": edge_data, "legend": legend_patches}

  def _plot(self, graph, layout, polygon_gdf, edge_styles, vertex_color, figsize, legend_patches,
            add_basemap=False, basemap_provider=None, basemap_attribution="", basemap_attribution_size=6,
            node_size=1, polygon_edgecolor="lightgrey", polygon_linewidth=0.5, polygon_category=None,
            polygon_cmap="tab20", polygon_alpha=0.5,
            legend_title="Edge weights (Jenks)", zoom_to_nodes=False, title=None,
            health_centre_df=None, health_centre_id_col="TractID", health_centre_value_col="n_visits",
            health_node_k_classes=5, health_node_color="dodgerblue",
            health_node_size_min=20, health_node_size_max=100, health_node_alpha=1.0):

    self.fig, self.ax = plt.subplots(figsize=figsize)
    ax = self.ax
 
    if title:
      ax.set_title(title, fontsize=12)

    if polygon_gdf is not None:
      if polygon_category is not None and polygon_category in polygon_gdf.columns:
        all_categories = sorted(polygon_gdf[polygon_category].dropna().unique())
        cmap = plt.get_cmap(polygon_cmap)
        color_dict = {cat: cmap(i % cmap.N) for i, cat in enumerate(all_categories)}
        polygon_gdf['color'] = polygon_gdf[polygon_category].map(color_dict)

        polygon_gdf.plot(ax=ax,
                        facecolor=polygon_gdf["color"],
                        edgecolor=polygon_edgecolor,
                        alpha=polygon_alpha,
                        linewidth=polygon_linewidth,
                        zorder=1)
      else:
        polygon_gdf.boundary.plot(ax=ax,
                                  edgecolor=polygon_edgecolor,
                                  linewidth=polygon_linewidth,
                                  zorder=1)

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
    node_ids = list(graph.nodes())
    node_sizes = np.full(len(node_ids), node_size, dtype=float)
    node_colors = np.array([to_rgba(vertex_color)] * len(node_ids))
    node_alphas = np.ones(len(node_ids), dtype=float)

    if health_centre_df is not None and not health_centre_df.empty:
      health_centre_df[health_centre_id_col] = health_centre_df[health_centre_id_col].astype(str)
      id_to_index = {n: i for i, n in enumerate(node_ids)}
      values = health_centre_df[health_centre_value_col].values
      classifier = mapclassify.NaturalBreaks(values, k=health_node_k_classes)
      labels = classifier.yb
      size_range = health_node_size_max - health_node_size_min
      scaled_sizes = np.array([
        health_node_size_min + (i / (health_node_k_classes - 1)) ** 2 * size_range
        for i in range(health_node_k_classes)
      ])
      for node, cls in zip(health_centre_df[health_centre_id_col], labels):
        if node in id_to_index:
          idx = id_to_index[node]
          node_sizes[idx] = scaled_sizes[cls]
          node_colors[idx] = to_rgba(health_node_color, health_node_alpha)
          node_alphas[idx] = health_node_alpha

    if np.unique(node_alphas).size == 1:
      ax.scatter(xs, ys, s=node_sizes, c=node_colors, alpha=node_alphas[0], zorder=3, edgecolors='none')
    else:
      rgba_colors = [to_rgba(c, a) for c, a in zip(node_colors, node_alphas)]
      ax.scatter(xs, ys, s=node_sizes, c=rgba_colors, zorder=3, edgecolors='none')

    if zoom_to_nodes:
      margin = 100
      ax.set_xlim(xs.min() - margin, xs.max() + margin)
      ax.set_ylim(ys.min() - margin, ys.max() + margin)

    if legend_patches:
      ax.legend(
        handles=legend_patches,
        title=legend_title,
        loc="center right",
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

  def save_plot(self, path, dpi=300, bbox_inches="tight", transparent=True):
    if self.fig:
      self.fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent)
    else:
      raise RuntimeError("No figure to save. Run plot_graph() first.")
