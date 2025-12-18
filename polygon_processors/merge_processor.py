import pandas as pd
import geopandas as gpd
from pathlib import Path
from .base_processor import PolygonProcessor

class UrbanRuralPolygonMerger(PolygonProcessor):
    """
    Processor for merging urban/rural polygons

    Args:
        list_coms (list[int]): List of commune codes to process.
            Defaults to the common communes in the southeastern sector of the Metropolitan Region.
        poly_id (str or None): Column name for polygon IDs.
        num_cols (list[int]): List of numeric columns to include.
    """
    
    def __init__(self, list_coms=None, poly_id="block_id", num_cols=["n_per", "n_vp_ocupada"]):
        self.list_coms = list_coms or [13110, 13111, 13112, 13202, 13201, 13131, 13203]
        self.poly_id = poly_id
        self.num_cols = num_cols
        
    def process(self, 
                urban_path="data/raw/Cartografía_censo2024_R13_gdf.parquet",
                rural_path="data/raw/Cartografía_censo2024_R13_Entidades.parquet"):
        """
        Process and merge data from specified paths.
        
        Args:
            urban_path (str or Path): Path to urban blocks ZIP/shapefile
            rural_path (str or Path): Path to rural entities ZIP/shapefile
            
        Returns:
            GeoDataFrame: Merged polygons in EPSG:32719
        """
        urban_path = Path(urban_path).absolute()
        rural_path = Path(rural_path).absolute()
    
        if not urban_path.exists():
            raise FileNotFoundError(f"Urban data not found: {urban_path}")
        
        if not rural_path.exists():
            raise FileNotFoundError(f"Rural data not found: {rural_path}")
    
        urban_blocks = self._load_and_process(
            urban_path,
            urban=True
        )
        rural_entities = self._load_and_process(
            rural_path,
            urban=False
        )

        # Create unary union of rural geometries
        rural_union = rural_entities.unary_union

        # Subtract rural areas from urban polygons, in case they intersect
        urban_blocks["geometry"] = urban_blocks.geometry.difference(rural_union)

        urban_blocks, _ = self.identify_multipart_polygons(
            urban_blocks, self.poly_id, keep_largest=True)

        return pd.concat([urban_blocks, rural_entities], axis=0)
    
    def _load_and_process(self, path, urban=True):
        """"""
        gdf = gpd.read_parquet(path)

        gdf["geometry"] = gdf.geometry
        gdf = gdf.set_geometry("geometry")
        
        gdf = gdf.rename(columns={"MANZENT": self.poly_id, 
                                    "COMUNA": "commune",
                                    "CUT": "commune_id"})
            
        gdf["commune_id"] = gdf["commune_id"].astype(int)
        gdf = gdf.loc[gdf["commune_id"].isin(self.list_coms)]

        # Filter polygons without population
        gdf = gdf.loc[gdf["n_per"]>0]

        gdf = gdf[["commune_id", "commune", self.poly_id, "geometry"] + self.num_cols]

        if urban:
            zone_type = "Urban"
        else:
            zone_type = "Rural"

        gdf["zone_type"] = zone_type
        gdf = self._validate_crs(gdf)
            
        return gdf