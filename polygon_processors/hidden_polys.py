import geopandas as gpd
from rtree import index

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
    
    @staticmethod
    def find_partial_overlaps(gdf, min_overlap_area=1.0):
        """
        Find polygons that partially overlap with others.
        
        Args:
            gdf (GeoDataFrame): Input polygons to check
            min_overlap_area (float): Minimum overlap area to consider
            
        Returns:
            tuple: (overlap_gdf, overlap_indices)
        """
    
        # Create spatial index
        idx = index.Index()
        for i, geom in enumerate(gdf.geometry):
            idx.insert(i, geom.bounds)
    
        overlap_indices = set()
        overlapping_pairs = []
        processed_pairs = set()  # To track which pairs we've handled

        # First pass: identify all overlaps
        for i, (_, row) in enumerate(gdf.iterrows()):
            candidates = list(idx.intersection(row.geometry.bounds))
        
            for j in candidates:
                if i >= j:  # Avoid duplicate checks
                    continue
                other_geom = gdf.iloc[j].geometry
                intersection = row.geometry.intersection(other_geom)
            
                if (not intersection.is_empty and 
                    intersection.area >= min_overlap_area and
                    not row.geometry.equals(other_geom)):
                
                    overlap_indices.add(i)
                    overlap_indices.add(j)
                    if (j, i) not in processed_pairs:  # Ensure we don't process reverse pairs
                        overlapping_pairs.append((i, j, intersection))
                        processed_pairs.add((i, j))

        # Create a copy to modify
        modified_gdf = gdf.copy()
    
        # Process each overlapping pair to distribute overlaps
        for i, j, intersection in overlapping_pairs:
            geom_i = modified_gdf.loc[i, 'geometry']
            geom_j = modified_gdf.loc[j, 'geometry']
        
            # Decision rule: keep overlap in the smaller polygon
            if geom_i.area < geom_j.area:
                # Keep overlap in i, remove from j
                modified_gdf.loc[j, 'geometry'] = geom_j.difference(intersection)
            else:
                # Keep overlap in j, remove from i
                modified_gdf.loc[i, 'geometry'] = geom_i.difference(intersection)
    
        # Get only the modified overlapping geometries
        overlap_gdf = modified_gdf.iloc[list(overlap_indices)].copy()
    
        return overlap_gdf, list(overlap_indices)