# GIS Dissertation

This repository contains the data, code, and documentation for my MSc dissertation titled:  
**"A Proposal for Developing Primary Care Service Areas in the South-Eastern Metropolitan Health Service of Chile"**


## Example Workflow

<details>
  <summary><strong> Step 1: Combine, standardise census chilean polygons and process multipart polygons </strong></summary>

  Example command
  ```python
  python multipart_processing.py -u 'manzanas_apc_2023.shp' -r 'microdatos_entidad.zip' -o 'processed_data.shp'
 ```
- -u: Path to the input urban census polygons data file

- -r: Path to the input rural census polygons data file

- -o: Output datafile with processed census polygons
</details>

<details>
  <summary><strong> Step 2: Standardise census chilean polygons and process multipart polygons </strong></summary>

  Example command 
  ```python
  python voronoi_polys.py -i 'processed_data.shp' -r 'COMUNA_C17.shp' -b 'hidrographic_network.shp' -o 'voronoi.gpkg'
 ```
- -i: Input processed polygons shapefile

- -r: Region boundary shapefile

- -b: Barrier layer shapefile (e.g., hydrographic network)

- -o: Output Voronoi GeoPackage file
</details>
