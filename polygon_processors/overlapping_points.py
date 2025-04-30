import numpy as np
from shapely.geometry import Point
from tqdm import tqdm

class OverlappingPointHandler:
    """Handles overlapping points by jittering them within constrained buffer zones."""

    @staticmethod
    def move_point_within_buffer(point, polygon, buffer_radius=10):
        """Move a single point randomly within a buffer clipped by a polygon."""
        buffer = point.buffer(buffer_radius)
        clipped_area = buffer.intersection(polygon)
    
        if clipped_area.is_empty:
            return point  # If nothing left, keep original

        minx, miny, maxx, maxy = clipped_area.bounds

        for _ in range(30):  # Try 30 times
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            candidate = Point(x, y)
            if clipped_area.contains(candidate):
                return candidate

        return point
    

    @staticmethod
    def move_overlapping_points(gdf_points, gdf_polygons, buffer_radius=10, overlap_threshold=10):
        """Move points that have ≥ threshold overlapping lon/lat coordinates."""
        # Make sure CRS are aligned
        assert gdf_points.crs == gdf_polygons.crs, "CRS must match!"
    
        # Create longitude/latitude columns
        gdf_points['lon'] = gdf_points.geometry.x
        gdf_points['lat'] = gdf_points.geometry.y

        # Group by (lon, lat)
        grouped = gdf_points.groupby(['lon', 'lat'])

        # Count groups with overlaps
        overlapping_count = sum(len(group) >= overlap_threshold for _, group in grouped)
        print(f"Found {overlapping_count} location(s) with ≥ {overlap_threshold} overlapping points.")

        new_geometries = []

        for (lon, lat), group in tqdm(grouped, desc="Processing overlapping points"):
            if len(group) >= overlap_threshold:
                # Points to move
                for idx, row in group.iterrows():
                    # Find the polygon that contains the point
                    containing_polygons = gdf_polygons[gdf_polygons.contains(row.geometry)]
                
                    if containing_polygons.empty:
                        # No polygon found, keep original
                        new_geometries.append(row.geometry)
                        continue
                
                    # Take first polygon
                    polygon = containing_polygons.iloc[0].geometry

                    # Move point safely
                    moved_point = OverlappingPointHandler.move_point_within_buffer(row.geometry, polygon, buffer_radius)
                    new_geometries.append(moved_point)
            else:
                # Points without overlaps (leave them as they are)
                for idx, row in group.iterrows():
                    new_geometries.append(row.geometry)

        # Update the geometries
        gdf_points = gdf_points.drop(columns=['lon', 'lat']).copy()
        gdf_points.geometry = new_geometries

        return gdf_points
