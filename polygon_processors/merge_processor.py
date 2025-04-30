import pandas as pd
import geopandas as gpd
from pathlib import Path
from .base_processor import PolygonProcessor

class UrbanRuralPolygonMerger(PolygonProcessor):
    """
    Simplified processor for merging urban/rural polygons without root_folder dependency.
    Works with direct file paths.
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
                manzanas_path="polygon_processors/data/manzanas_apc_2023.shp",
                entidades_path="polygon_processors/data/microdatos_entidad.zip"):
        """
        Process and merge data from specified paths.
        
        Args:
            manzanas_path (str|Path): Path to urban blocks ZIP/shapefile
            entidades_path (str|Path): Path to rural entities ZIP/shapefile
            
        Returns:
            GeoDataFrame: Merged polygons in EPSG:32719
        """
        manzanas_path = Path(manzanas_path).absolute()
        entidades_path = Path(entidades_path).absolute()
    
        if not manzanas_path.exists():
            raise FileNotFoundError(f"Urban data not found: {manzanas_path}")
    
        manzanas = self._load_and_process(
            manzanas_path,
            urban=True
        )
        entidades = self._load_and_process(
            entidades_path,
            urban=False
        )
        return pd.concat([manzanas, entidades], axis=0)
    
    def _load_and_process(self, path, urban=True):
        """Unified loader for both data types"""
        gdf = gpd.read_file(path)
        
        if urban:
            gdf = gdf.drop(columns="MANZENT")
            gdf = gdf.rename(columns={"Mzent_TX": "MANZENT", "N_COMUNA": "COMUNA"})
            gdf = gdf.loc[gdf["CUT"].isin(self.list_coms)]
            gdf = gdf.loc[gdf["VIVIENDA"]>0]
            gdf = gdf[["CUT", "COMUNA", "MANZENT", "geometry"]]
            gdf["TIPO_ZONA"] = "MANZANA"
            gdf = self._validate_crs(gdf)
        else:
            gdf = gdf.rename(columns={"COD_COMUNA": "CUT", "NOMBRE_COM": "COMUNA"})
            gdf["CUT"] = gdf["CUT"].astype(int)
            gdf = gdf.loc[gdf["CUT"].isin(self.list_coms)]
            gdf["DISTRITO"] = gdf["DISTRITO"].astype(int)
            gdf["MANZENT"] = (
                gdf["CUT"].astype(str) + 
                gdf["DISTRITO"].astype(str).str.zfill(2) +
                "2" + 
                gdf["CODIGO_LOC"].str.zfill(3) + 
                gdf["CODIGO_ENT"].str.zfill(3)
            )
            gdf = gdf[["CUT", "COMUNA", "MANZENT", "geometry"]]
            gdf["TIPO_ZONA"] = "ENTIDAD"
            gdf = self._validate_crs(gdf)
            
        return gdf