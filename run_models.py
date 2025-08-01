from community_detection import CommunityDetectionBatcher
from pathlib import Path
from paths import get_paths
import argparse
import time
import json

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Community Detection Pipeline\n"
                    "Runs multiple community detection algorithms across different parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/output paths
    parser.add_argument('-t', '--tracts', type=str, default="tracts.gpkg",
                        help="Input tracts GeoPackage file")
    parser.add_argument('-p', '--patient-data', type=str, default="phc_consultations_2023.parquet",
                        help="Patient health consultation data CSV file")
    parser.add_argument('-l', '--locations', type=str, default="moved_points.parquet",
                        help="Processed patient locations Shapefile")
    parser.add_argument('-hc', '--health-centres', type=str, 
                        default="Establecimientos DEIS MINSAL 29-04-2025.xlsx",
                        help="Health centres Excel file")
    
    # Algorithm parameters
    parser.add_argument('--pop-values', type=int, nargs='+', 
                        default=[200, 300, 400, 500, 600, 700, 800,
                                 900, 1000, 1100, 1200, 1300, 1400, 1500],
                        help="Population thresholds to test")
    parser.add_argument('--weight-cols', type=str, nargs='+',
                        default=["n_visits", "visit_share", "combined_score"],
                        help="Weight columns to use for community detection")
    
    # Range parameters (start, end, step)
    parser.add_argument('--modules-range', type=float, nargs=3,
                        metavar=('START', 'END', 'STEP'),
                        default=[27, 44, 1],
                        help="Range of preferred modules for Infomap (start, end, step)")

    parser.add_argument('--resolutions-range', type=float, nargs=3,
                        metavar=('START', 'END', 'STEP'),
                        default=[3.5, 13.6, 0.1],
                        help="Range of resolutions for Louvain/Leiden (start, end, step)")
    
    # Performance options
    parser.add_argument('-j', '--jobs', type=int, default=12,
                        help="Number of parallel jobs to run")
    parser.add_argument('--trials', type=int, default=1,
                        help="Number of trials for each algorithm run")
    
    # Output control
    parser.add_argument('--save-config', action='store_true',
                        help="Save the configuration to a JSON file")

    # Spatial configs as JSON string (optional)
    parser.add_argument('--spatial-configs', type=str, default=None,
                        help="JSON string for spatial configs list [{'enforce_spatial': bool, 'strategy': str or None}, ...]")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    # Set up directories
    input_dir, output_dir, aztool_dir = get_paths()
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default spatial configs if none provided
    default_spatial_configs = [
        {"enforce_spatial": False, "strategy": None},
        {"enforce_spatial": True, "strategy": "strongest_connection"},
        {"enforce_spatial": True, "strategy": "min_impact_score"},
    ]

    # Parse spatial configs if provided
    if args.spatial_configs:
        try:
            spatial_configs = json.loads(args.spatial_configs)
            if not isinstance(spatial_configs, list):
                raise ValueError("spatial-configs must be a JSON list")
        except Exception as e:
            print(f"Error parsing spatial-configs JSON: {e}")
            print("Falling back to default spatial configurations")
            spatial_configs = default_spatial_configs
    else:
        spatial_configs = default_spatial_configs

    # Build configuration
    config = {
        "pop_values": args.pop_values,
        "weight_cols": args.weight_cols,
        "modules_range": tuple(args.modules_range),
        "resolutions_range": tuple(args.resolutions_range),
        "trials": args.trials,
        "file_paths": {
            "tracts": str(output_dir / args.tracts),
            "patient_data": str(input_dir / args.patient_data),
            "locations": str(output_dir / args.locations),
            "health_centres": str(input_dir / args.health_centres),
            #'matrices': str(output_dir / "all_matrices.csv"),
            #"flows": str(output_dir / "flows.gpkg"),
            #"community_assignments": str(output_dir / "tracts_community_assignments.csv"),
            #"communities": str(output_dir / "tracts_communities.gpkg"),
            #"li_results": str(output_dir / "combined_li_results_all.xlsx"),
            "summary_stats": str(output_dir / "li_summary_stats_all.xlsx")
        },
        "spatial_configs": spatial_configs,
    }
    
    # Save config if requested
    if args.save_config:
        config_path = output_dir / "config_parameters.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved configuration to {config_path}")

    # Run pipeline
    start_time = time.time()
    print("Starting community detection pipeline...")

    batcher = CommunityDetectionBatcher(config)

    batcher.spatial_configs = spatial_configs

    batcher.run(n_jobs=args.jobs)

    duration_minutes = (time.time() - start_time) / 60
    print(f"\nPipeline completed in {duration_minutes:.2f} minutes")