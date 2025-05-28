import pandas as pd
import geopandas as gpd
from pathlib import Path
from .base_processor import PolygonProcessor

class UrbanRuralPolygonMerger(PolygonProcessor):
    """
    Processor for merging urban/rural polygons
    """
    
    def __init__(self, list_coms=None):
        """
        Initialise with commune codes to include.
        
        Args:
            list_coms (list[int]): List of commune codes to process.
                Defaults to common communes in Santiago suroriente.
        """
        self.list_coms = list_coms or [13110, 13111, 13112, 13202, 13201, 13131, 13203]
        
    def process(self, 
                urban_path="data/raw/manzanas_apc_2023.shp",
                rural_path="data/raw/microdatos_entidad.zip"):
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

        return pd.concat([urban_blocks, rural_entities], axis=0)
    
    def _load_and_process(self, path, urban=True):
        """Unified loader for both data types"""
        gdf = gpd.read_file(path)
        
        if urban:
            gdf = gdf.rename(columns={"Mzent_TX": "block_id", 
                                      "N_COMUNA": "commune",
                                      "CUT": "commune_id"})
            
            gdf["commune_id"] = gdf["commune_id"].astype(int)
            gdf = gdf.loc[gdf["commune_id"].isin(self.list_coms)]

            # Filter polygons without residential housing
            gdf = gdf.loc[gdf["VIVIENDA"]>0]

            gdf = gdf[["commune_id", "commune", "block_id", "geometry"]]
            gdf["zone_type"] = "Urban"
            gdf = self._validate_crs(gdf)
        else:
            gdf = gdf.rename(columns={"COD_COMUNA": "commune_id",
                                      "NOMBRE_COM": "commune"})
            
            gdf["commune_id"] = gdf["commune_id"].astype(int)
            gdf = gdf.loc[gdf["commune_id"].isin(self.list_coms)]

            # Create unique code
            gdf["DISTRITO"] = gdf["DISTRITO"].astype(int)

            gdf["block_id"] = (
                gdf["commune_id"].astype(str) + 
                gdf["DISTRITO"].astype(str).str.zfill(2) +
                "2" + 
                gdf["CODIGO_LOC"].str.zfill(3) + 
                gdf["CODIGO_ENT"].str.zfill(3)
            )
            gdf = gdf[["commune_id", "commune", "block_id", "geometry"]]
            gdf["zone_type"] = "Rural"
            gdf = self._validate_crs(gdf)
            
        return gdf