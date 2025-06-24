# GIS Dissertation

This repository contains the data, code, and documentation for my MSc dissertation titled:  
**""**


## Example Workflow

<details>
  <summary><strong> Step 1: Combine, standardise census chilean polygons and process multipart polygons </strong></summary>

  Default command
  ```python
  python multipart_processing.py -u manzanas_apc_2023.shp -r microdatos_entidad.zip -o processed_polygons.shp
 ```
- `-u`: Path to the input urban census polygons data file

- `-r`: Path to the input rural census polygons data file

- `-o`: Output datafile with processed census polygons
</details>

<details>
  <summary><strong> Step 2: Generate Voronoi polygons constrained by regional boundaries </strong></summary>

  Default command:

  By default, the region is divided into chunks using intermediate regional boundaries. This partitioning enables the creation and processing of Voronoi diagrams in parallel, improving computational efficiency.
  ```python
  python voronoi_polys.py -i processed_polygons.shp -r COMUNA_C17.shp
 ```
- `-i`: Input processed polygons shapefile from the previous step

- `-r`: Region boundary shapefile

- `-b`: Barrier layer shapefile (e.g., hydrographic network)

- `-o`: Output Voronoi GeoPackage file

*Note*: Each Voronoi polygon set for a region is stored as a separate layer within the GeoPackage file.
Additionally, a combined Voronoi layer that merges all specified regions is saved in the same GeoPackage under the layer name `"combined"`. The same behaviour applies to hidden polygons detected during the Voronoi generation process: they are exported as separate layers and also merged into a combined layer within a separate GeoPackage file.

__Additional arguments:__

To disable specific behaviours—such as processing regions in chunks—and explore other options, run:

 ```python
  python voronoi_polys.py --help
 ```
For more advanced fine-tuning, including parameters like buffer sizes and boundary simplification tolerance levels, refer directly to the class implementation in [`voronoi_processor.py`](polygon_processors/voronoi_processor.py). Examples of visual outputs generated with various parameter settings can be found in the [`notebooks`](notebooks) directory.

</details>

<details>
  <summary><strong> Step 3: Calculate Voronoi attributes</strong></summary>
  This step generates the population estimates and socioeconomic group counts for each polygon based on point locations. These processed attributes and polygons serve as input for the AZTool software, supporting its aggregation process by enabling constraints that promote a certain level of socioeconomic homogeneity.

  __Default command__

  It is recommended to start by running the following command to visualise the population distribution, as some Voronoi polygons may contain significantly higher populations than others.
  ```python
  python calculate_attributes.py -vi combined -pi phc_consultations_2023.csv
 ```
- `-vi`: Path to input file containing Voronoi polygons (a layer within a GeoPackage)

- `-pi`: Input data with latitude and longitude columns (CSV or spatial format)

__Splitting High-Population Polygons__

To automatically split polygons that exceed a predefined population threshold, use the `--split-polygons` flag:

  ```python
  python calculate_attributes.py -vi combined -pi phc_consultations_2023.csv --split-polygons
  ```
When this option is enabled, the script applies a method that slightly shifts overlapping points within a specified distance buffer. This helps minimise artefacts caused by high-density buildings or stacked population points, while ensuring that the shifted points remain within their original containing polygon.

__Output Files__

By default, the output files are saved in Shapefile format, as required by the AZTool software. Depending on whether polygon splitting is enabled, the main output will be:

- `voronoi_data.shp` – when no splitting is applied

- `voronoi_data_split.shp` – when the `--split-polygons` option is used

In addition, the script also generates Shapefiles containing the point data used for the estimation process. These files help verify which population points were considered in the analysis.

__Additional arguments__

To view all available parameters—including the default population threshold for splitting and paths to socioeconomic datasets—run:

 ```python
  python calculate_attributes.py --help
 ```

</details>

<details>
  <summary><strong> Step 4: Create Tracts</strong></summary>

This step requires downloading the `AZtool (version 1.0.3 25/8/11)` and `AZImporter (version 1.0.1 20/10/10)` software from the [`oficial website`](https://aztool.geodata.soton.ac.uk/download/). Place each tool's directory within your project folder.

Then follow these steps:
1. Run the `AZImport.exe` file from the `AZImporter` directory.
 Use either `voronoi_data.shp` or `voronoi_data_split.shp` as input, and set the output to `voronoi.aat`, saving it within the `AZTool` directory.

2. 

  __Default command__ 
  ```python
  python create_tracts.py -i voronoi_data_split.shp -azt voronoi.pat
 ```
- `-i`: Path to the input file containing processed Voronoi polygons used as building blocks.

- `-azt`: Filename of AZTool Building Block IDs.

- `-o`: Output GeoPackage filename containing the resulting Tracts.

</details>