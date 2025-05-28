# GIS Dissertation

This repository contains the data, code, and documentation for my MSc dissertation titled:  
**"A Proposal for Developing Primary Care Service Areas in the South-Eastern Metropolitan Health Service of Chile"**


## Example Workflow

<details>
  <summary><strong> Step 1: Combine, standardise census chilean polygons and process multipart polygons </strong></summary>

  Default command
  ```python
  python multipart_processing.py -u 'manzanas_apc_2023.shp' -r 'microdatos_entidad.zip' -o 'processed_data.shp'
 ```
- `-u`: Path to the input urban census polygons data file

- `-r`: Path to the input rural census polygons data file

- `-o`: Output datafile with processed census polygons
</details>

<details>
  <summary><strong> Step 2: Generate Voronoi polygons constrained by regional boundaries </strong></summary>

  Default command
  ```python
  python voronoi_polys.py -i 'processed_data.shp' -r 'COMUNA_C17.shp' -b 'hidrographic_network.shp' -o 'voronoi.gpkg'
 ```
- `-i`: Input processed polygons shapefile

- `-r`: Region boundary shapefile

- `-b`: Barrier layer shapefile (e.g., hydrographic network)

- `-o`: Output Voronoi GeoPackage file

__Additional optional arguments:__

- `-l, --region_list`: Specify a list of region codes to process.
Defaults to `[13111, 13110, 13112, 13202, 13201, 13131, 13203]`, which correspond to the study area.
Example: `-l 13111 13201`

- `--overlay-hidden`: If enabled, performs an overlay operation between visible and hidden polygons (useful for advanced spatial processing).
Use flag only: `--overlay-hidden`

Additional parameters, such as buffer sizes and tolerance levels for boundary simplification, are available for fine-tuning the Voronoi polygon generation. For a complete list of options and detailed usage, please refer directly to the class code in `voronoi_processor.py`.

</details>
