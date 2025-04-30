from shapely.geometry import Polygon, MultiPolygon, LineString, LinearRing
import numpy as np

class PolygonDensifier:
    """Handles densification of polygons by adding vertices."""
    
    @staticmethod
    def densify_linestring(linestring, distance):
        """Densify a LineString by adding vertices at intervals."""
        num_vertices = int(np.ceil(linestring.length / distance))
        new_points = [linestring.interpolate(i * distance) for i in range(num_vertices + 1)]
        return LineString(new_points)
    
    @staticmethod
    def densify_polygon(polygon, distance):
        """Add vertices to a polygon at given intervals."""
        if polygon.is_empty:
            return polygon
            
        if polygon.geom_type == 'Polygon':
            lines = [LineString(ring) for ring in [polygon.exterior.coords, *polygon.interiors]]
            densified_rings = []
            
            for line in lines:
                try:
                    densified_line = PolygonDensifier.densify_linestring(line, distance)
                    if len(densified_line.coords) >= 4:
                        densified_rings.append(LinearRing(densified_line.coords))
                except Exception as e:
                    print(f"Error processing ring: {e}. Skipping invalid ring.")
            
            if not densified_rings:
                return polygon
                
            return Polygon(densified_rings[0], densified_rings[1:])

        elif polygon.geom_type == 'MultiPolygon':
            densified_polys = [PolygonDensifier.densify_polygon(poly, distance) for poly in polygon.geoms]
            return MultiPolygon([poly for poly in densified_polys if not poly.is_empty])

        return polygon
    
    @staticmethod
    def densify_geodataframe(gdf, distance):
        """Apply densification to all polygons in a GeoDataFrame."""
        gdf['geometry'] = gdf['geometry'].apply(
            lambda geom: PolygonDensifier.densify_polygon(geom, distance) if geom.is_valid else geom)
        return gdf