import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point

class PolygonPlotter:
    """
    Utility class to plot polygons
    """

    def __init__(self, data):
        """
        Initialise the plotter with a GeoDataFrame.
    
        Args:
            data (GeoDataFrame or str): A GeoDataFrame or file path to a geospatial dataset.
        """
        if isinstance(data, gpd.GeoDataFrame):
            self.gdf = data
        elif isinstance(data, str):
            self.gdf = gpd.read_file(data)
        else:
            raise TypeError("Input must be a GeoDataFrame or a valid file path.")

    def plot_target_with_context(
        self,
        target_gdf,
        bbox=None,
        pad=50,
        target_color='cyan',
        others_color='black',
        alpha_target=0.5,
        alpha_others=0.5,
        plot_boundary_points=False,
        boundary_point_size=1,
        boundary_point_color='yellow'
    ):
        """
        Plot target and other polygons with optional boundary points on a basemap.

        Args:
            target_gdf (GeoDataFrame): Subset of self.gdf representing target polygon(s).
            bbox (tuple, optional): (minx, miny, maxx, maxy). If None, uses target bounds.
            pad (float): Padding around bbox.
            target_color (str): Fill color for target polygon(s).
            others_color (str): Fill color for other polygons.
            alpha_target (float): Alpha for target polygon(s).
            alpha_others (float): Alpha for other polygons.
            plot_boundary_points (bool): If True, plots polygon boundaries as points.
            boundary_point_size (int): Marker size for boundary points.
            boundary_point_color (str): Color for boundary points.
        """
        others = self.gdf.drop(target_gdf.index)

        fig, ax = plt.subplots()

        target_gdf.plot(ax=ax, color=target_color, edgecolor='white', alpha=alpha_target)
        others.plot(ax=ax, color=others_color, edgecolor='white', alpha=alpha_others)

        if plot_boundary_points:
            boundary_points = []
            for geom in self.gdf.geometry:
                if geom.geom_type == 'Polygon':
                    boundary_points.extend(geom.exterior.coords)
                elif geom.geom_type == 'MultiPolygon':
                    for part in geom.geoms:
                        boundary_points.extend(part.exterior.coords)

            point_gdf = gpd.GeoDataFrame(geometry=[Point(xy) for xy in boundary_points], crs=self.gdf.crs)
            point_gdf.plot(ax=ax, color=boundary_point_color, markersize=boundary_point_size, alpha=1)

        if bbox is None:
            minx, miny, maxx, maxy = target_gdf.total_bounds
        else:
            minx, miny, maxx, maxy = bbox

        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)

        ctx.add_basemap(ax,
                        source=ctx.providers.Esri.WorldImagery,
                        crs=self.gdf.crs.to_string(),
                        attribution_size=6,
                        attribution="Tiles (C) Esri")

        ax.set_axis_off()
        plt.show()