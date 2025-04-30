import geopandas as gpd

class HiddenPolygonProcessor:
    """Handles identification and processing of hidden polygons."""
    
    @staticmethod
    def find_hidden_polygons(gdf):
    # Create an empty list to store polygons that are behind others
        hidden_polygons = []

        # Loop through all pairs of polygons
        for idx, poly in gdf.iterrows():
            for _, other_poly in gdf.iterrows():
            # Compute the difference between the polygons
                if poly['geometry'].intersects(other_poly['geometry']) and poly['geometry'] != other_poly['geometry']:
                    diff = poly['geometry'].difference(other_poly['geometry'])
                    if diff.is_empty:
                        hidden_polygons.append(poly)

        # Create a GeoDataFrame with polygons that are behind others
        hidden_gdf = gpd.GeoDataFrame(hidden_polygons, columns=['geometry'])

        # Set the same CRS as the original GeoDataFrame
        hidden_gdf.crs = gdf.crs

        # Extract the indices of the polygons that are hidden
        hidden_indices = hidden_gdf.index

        return hidden_gdf, hidden_indices