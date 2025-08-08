import pandas as pd
import geopandas as gpd
import os
from typing import Dict
from joblib import Parallel, delayed
from .create_matrix import MatrixConstructor
from .create_graph import GraphConstructor
from .communities import CommunityDetector
from .metrics import CommunityMetrics

class CommunityDetectionBatcher:
    """Batch runner for community detection experiments over multiple configurations.

    Runs multiple community detection algorithms and parameter settings
    on different spatial and population configurations, collects results,
    and exports summaries.
    """
    def __init__(self, config: Dict, input_matrix: pd.DataFrame = None):
        """Initialises the batcher with a configuration dictionary.

        Args:
            config (Dict): Configuration dictionary with keys such as
                'trials', 'spatial_configs', 'pop_values', 'weight_cols',
                'modules_range', 'resolutions_range', and 'file_paths'.

                - `pop_values` is a list of integers representing the average population
                  size for each polygon/tract configuration (referred to as `target_pop`).
        """
        self.input_matrix = input_matrix
        self.trials = config.get('trials', 30)
        self.spatial_configs = config.get('spatial_configs', [
            {"enforce_spatial": False, "strategy": None},
            {"enforce_spatial": True, "strategy": "min_impact_score"},
            {"enforce_spatial": True, "strategy": "strongest_connection"}
        ])
        self.pop_values = config.get('pop_values', [200, 300, 400, 500, 600, 700, 800,
                                                     900, 1000, 1100, 1200, 1300, 1400, 1500])
        self.weight_cols = config.get('weight_cols', ["n_visits", "visit_share", "combined_score"])
        self.modules_range = config.get('modules_range', (27, 44, 1))
        self.resolutions_range = config.get('resolutions_range', (3.5, 13.6, 0.1))


        if self.input_matrix is None:
            self.file_paths = config.get('file_paths', {
                'tracts': "data/processed/tracts.gpkg",
                'patient_data': "data/raw/data.csv",
                'locations': "data/processed/moved_points.shp",
                'health_centres': "data/raw/Establecimientos DEIS MINSAL 29-04-2025.xlsx",
                'matrices': "data/processed/all_matrices.csv",
                'flows': "data/processed/flows.gpkg",
                'community_assignments': "data/processed/tracts_community_assignments.csv",
                'li_results': "data/processed/combined_li_results_all.xlsx",
                'summary_stats': "data/processed/li_summary_stats_all.xlsx"
            })
        else:
            self.file_paths = config.get('file_paths', {
                'tracts': "data/processed/tracts.gpkg",
                'flows': "data/processed/flows.gpkg",
                'li_results': "data/processed/combined_li_results_all.xlsx",
                'summary_stats': "data/processed/li_summary_stats_all.xlsx"
            })

        self.all_li_results = []
        self.optimisation_scores = {}
        self.all_matrices = []
        self.all_community_assignments = []
        self.all_flows_list = []
        self.written_flows = set()
        self.execution_times = {}

    def _delete_gpkg_files_once(self):
        """Deletes the GeoPackage flows file if it exists to start fresh."""
        gpkg_path = self.file_paths.get("flows")
        if gpkg_path and os.path.exists(gpkg_path):
            try:
                os.remove(gpkg_path)
                print(f"Deleted file: {gpkg_path}")
            except Exception as e:
                print(f"Could not delete file {gpkg_path}: {e}")

    def run(self, n_jobs: int = -1):
        """Runs the batch community detection for all configured parameters in parallel.

        Args:
            n_jobs (int): Number of parallel jobs. Defaults to -1 (all cores).
        """
        self._delete_gpkg_files_once()

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._run_single_config)(pop, weight_col, sc.get("enforce_spatial", False), sc.get("strategy"))
            for pop in self.pop_values
            for weight_col in self.weight_cols
            for sc in self.spatial_configs
        )

        self._process_results(results)
        self._export_results()
        #self._write_geodataframes()

    def _process_results(self, results):
        """Processes results from parallel runs, aggregating outputs."""
        for matrix, community_df, li_results, opt_scores, flows, exec_times in results:
            #self.all_matrices.append(matrix)
            self.all_community_assignments.append(community_df)
            if not li_results.empty:
                self.all_li_results.append(li_results)
            self.optimisation_scores.update(opt_scores)
            #self.all_flows_list.append(flows)
            self.execution_times.update(exec_times)

    def _write_geodataframes(self):
        """Writes combined flow GeoDataFrames to GeoPackage layers by population."""
        all_flows = gpd.GeoDataFrame(pd.concat(self.all_flows_list, ignore_index=True))

        gpkg_path = self.file_paths['flows']
        for pop in all_flows['target_pop'].unique():
            if pop not in self.written_flows:
                subset = all_flows[all_flows['target_pop'] == pop].drop(columns=['layer_name'], errors='ignore')
                layer_name = f"flows_{pop}"
                subset.to_file(gpkg_path, layer=layer_name, driver='GPKG')
                self.written_flows.add(pop)

    def _run_single_config(self, pop, weight_col, enforce_spatial, strategy):
        """Runs community detection for a single parameter configuration.

        Args:
            pop (int): Average population size of polygons (tracts) in this configuration,
                       referred to as `target_pop`.
            weight_col (str): Weight column name in the matrix.
            enforce_spatial (bool): Whether to enforce spatial constraints.
            strategy (str): Strategy name or None.

        Returns:
            tuple: matrix, community_df, li_results, local_opt_scores, flows, execution_times
        """
        print(f"\n=== Processing pop: {pop} with weight: {weight_col} | Spatial: {enforce_spatial} ({strategy}) ===")
        self.execution_times = {}  # Reset execution times
        
        if self.input_matrix is not None:
            matrix = self.input_matrix[self.input_matrix["target_pop"] == pop].copy()
            matrix["weight_column"] = weight_col

            mc = MatrixConstructor()
            mc.matrix = matrix
            flows = mc.create_flow_lines()
            flows["target_pop"] = pop

            tracts = gpd.read_file(self.file_paths['tracts'], layer=f"tracts_{pop}")
            tracts["TractID"] = tracts["TractID"].astype(str)
        else:
            matrix, tracts, flows = self._construct_matrix(pop, weight_col)
        
        G = self._build_graph(matrix, weight_col)
        community_df, local_opt_scores, li_results = self._run_community_detection(
            G, matrix, tracts, pop, weight_col, enforce_spatial, strategy
        )
    
        return matrix, community_df, li_results, local_opt_scores, flows, self.execution_times

    def _append_community_result(self, res_dict, algorithm, param, cd, matrix, community_dfs,
                                  opt_scores, pop, weight_col, enforce_spatial, strategy,
                                  metric_value=None, exec_time=None):
        """Appends results from a single community detection run to internal accumulators."""
        df = pd.DataFrame(res_dict.items(), columns=["TractID", "community"])
        df["algorithm"] = algorithm
        df["param"] = param
        df["target_pop"] = pop
        df["weight_column"] = weight_col
        df["enforce_spatial"] = enforce_spatial

        strategy_value = strategy if strategy is not None else "none"
        df["strategy"] = strategy_value
        community_dfs.append(df)

        li = cd.compute_localisation_index(res_dict, matrix, weight_col="n_visits")
        li["algorithm"] = algorithm
        li["param"] = param
        li["target_pop"] = pop
        li["weight_column"] = weight_col
        li["enforce_spatial"] = enforce_spatial
        li["strategy"] = strategy_value
        self.all_li_results.append(li)

        key = (pop, weight_col, algorithm, param, enforce_spatial, strategy_value)
        if metric_value is not None:
            opt_scores[key] = metric_value
        if exec_time is not None:
            self.execution_times[key] = exec_time

    def _run_community_detection(self, G, matrix, tracts, pop: int, weight_col: str, enforce_spatial: bool, strategy: str):
        cd = CommunityDetector(G, dest_tracts=matrix['TractID_2'].unique(), tracts=tracts, trials=self.trials)
        community_dfs = []
        local_opt_scores = {}

        self._run_default_algorithms(cd, community_dfs, local_opt_scores, matrix, pop, weight_col, enforce_spatial, strategy)
        self._run_varying_modules_algorithms(cd, community_dfs, local_opt_scores, matrix, pop, weight_col, enforce_spatial, strategy)
        self._run_varying_resolution_algorithms(cd, community_dfs, local_opt_scores, matrix, pop, weight_col, enforce_spatial, strategy)

        community_df = pd.concat(community_dfs, axis=0)
        return community_df, local_opt_scores, pd.concat(self.all_li_results[-len(community_dfs):], axis=0)

    def _run_default_algorithms(self, cd, community_dfs, opt_scores, matrix, pop, weight_col, enforce_spatial, strategy):
        res, score, time = cd.run_infomap(return_codelength=True, enforce_spatial=enforce_spatial, strategy=strategy, measure_time=True)
        self._append_community_result(res, "infomap", "def", cd, matrix, community_dfs, opt_scores, pop, weight_col, enforce_spatial, strategy, score, time)

        res, score, time = cd.run_louvain(return_modularity=True, enforce_spatial=enforce_spatial, strategy=strategy, measure_time=True)
        self._append_community_result(res, "louvain", "def", cd, matrix, community_dfs, opt_scores, pop, weight_col, enforce_spatial, strategy, score, time)

        res, score, time = cd.run_leiden(return_modularity=True, enforce_spatial=enforce_spatial, strategy=strategy, measure_time=True)
        self._append_community_result(res, "leiden", "def", cd, matrix, community_dfs, opt_scores, pop, weight_col, enforce_spatial, strategy, score, time)

    def _run_varying_modules_algorithms(self, cd, community_dfs, opt_scores, matrix, pop, weight_col, enforce_spatial, strategy):
        start, end, step = self.modules_range
        for modules in range(start, end, step):
            res, score, time = cd.run_infomap(preferred_modules=modules, return_codelength=True, enforce_spatial=enforce_spatial, strategy=strategy, measure_time=True)
            self._append_community_result(res, "infomap", modules, cd, matrix, community_dfs, opt_scores, pop, weight_col, enforce_spatial, strategy, score, time)

    def _run_varying_resolution_algorithms(self, cd, community_dfs, opt_scores, matrix, pop, weight_col, enforce_spatial, strategy):
        start, end, step = self.resolutions_range
        num_steps = int(round((end - start) / step))
        resolutions = sorted(set(round(start + i * step, 1) for i in range(num_steps)))

        for res in resolutions:
            r1, s1, time = cd.run_louvain(resolution=res, return_modularity=True, enforce_spatial=enforce_spatial, strategy=strategy, measure_time=True)
            self._append_community_result(r1, "louvain", res, cd, matrix, community_dfs, opt_scores, pop, weight_col, enforce_spatial, strategy, s1, time)

            r2, s2, time = cd.run_leiden(resolution=res, return_modularity=True, enforce_spatial=enforce_spatial, strategy=strategy, measure_time=True)
            self._append_community_result(r2, "leiden", res, cd, matrix, community_dfs, opt_scores, pop, weight_col, enforce_spatial, strategy, s2, time)

    def _construct_matrix(self, pop: int, weight_col: str):
        tracts = gpd.read_file(self.file_paths['tracts'], layer=f"tracts_{pop}")
        
        matrix_builder = MatrixConstructor(
            tracts_gdf=tracts,
            patient_data_path=self.file_paths['patient_data'],
            locations_data_path=self.file_paths['locations'],
            health_centres_path=self.file_paths['health_centres']
        )
        matrix_builder.load_all_data()
        matrix_builder.compute_matrix()

        flows = matrix_builder.create_flow_lines()
        flows["target_pop"] = pop

        matrix = matrix_builder.matrix.copy()
        matrix["target_pop"] = pop
        matrix["weight_column"] = weight_col

        tracts = matrix_builder.tracts.copy()
        tracts["TractID"] = tracts["TractID"].astype(str)

        return matrix, tracts, flows

    def _build_graph(self, matrix: pd.DataFrame, weight_col: str):
        gc = GraphConstructor(matrix, weights_col=weight_col)
        gc.build_graph()
        return gc.get_graph()

    def _export_results(self):
        """Exports localisation index results and summary statistics to Excel files."""
        #if self.all_matrices:
        #    pd.concat(self.all_matrices, ignore_index=True).to_csv(self.file_paths['matrices'], index=False)

        #if self.all_community_assignments:
        #    pd.concat(self.all_community_assignments, ignore_index=True).to_csv(
        #        self.file_paths["community_assignments"], index=False)
        
        if self.all_li_results:
            combined_df = pd.concat(self.all_li_results, ignore_index=True)
            #combined_df.to_excel(self.file_paths['li_results'], index=False)
        
            opt_df = pd.DataFrame([
                {
                    "target_pop": pop,
                    "weight_column": weight_col,
                    "algorithm": algorithm,
                    "param": param,
                    "enforce_spatial": enforce_spatial,
                    "strategy": strategy,
                    "optimisation_score": score,
                    "execution_time": self.execution_times.get(
                        (pop, weight_col, algorithm, param, enforce_spatial, strategy)
                    )
                }
                for (pop, weight_col, algorithm, param, enforce_spatial, strategy), score in self.optimisation_scores.items()
            ])

            grouping_cols = ["target_pop", "weight_column", "algorithm", "param", "enforce_spatial", "strategy"]

            summary_stats_zero = CommunityMetrics.summarise_localisation_index(
                combined_df.copy(), exclude_zero_flows=False, groupby_cols=grouping_cols
            )
            
            summary_stats_non_zero = CommunityMetrics.summarise_localisation_index(
                combined_df.copy(), exclude_zero_flows=True, groupby_cols=grouping_cols
            )

            summary_stats = pd.concat([summary_stats_non_zero, summary_stats_zero], axis=0)
            summary_stats = summary_stats.merge(opt_df, on=grouping_cols, how="left")
            summary_stats.to_excel(self.file_paths['summary_stats'], index=False)

            print(f"\nExported {len(combined_df)} LI entries from all configurations.")
        else:
            print("\nNo localisation index results to export.")