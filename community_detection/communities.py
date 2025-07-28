import time
from infomap import Infomap
from .spatial_enforcer import SpatialContiguityEnforcer
from .metrics import CommunityMetrics

class CommunityDetector:
    """Detects communities in a spatial network using Louvain, Leiden, or Infomap algorithms.

    Supports optional spatial contiguity enforcement and metric computation for the resulting communities.

    Args:
        G (igraph.Graph): Input graph with 'TractID' vertex attribute and weighted edges.
        tracts (GeoDataFrame, optional): GeoDataFrame of spatial units for enforcing contiguity.
        dest_tracts (GeoDataFrame, optional): Optional GeoDataFrame used for advanced metrics.
        seed (int): Random seed for reproducibility.
        trials (int): Number of trials for community detection algorithms that support it.
    """

    def __init__(self, G, tracts=None, dest_tracts=None, seed=42, trials=1):
        self.g_ig = G
        self.tracts = tracts
        self.dest_tracts = dest_tracts
        self.seed = seed
        self.trials = trials

        if "TractID" not in self.g_ig.vs.attributes():
            raise ValueError("Graph vertices must have a 'TractID' attribute")

        self.g_ig.vs["name"] = self.g_ig.vs["TractID"]

    def run_louvain(self, resolution=None, return_modularity=False,
                    enforce_spatial=False, score_metric='modularity',
                    strategy='min_impact_score', verbose=False,
                    measure_time=False):
        """Runs Louvain community detection.

        Args:
            resolution (float, optional): Resolution parameter for modularity optimization.
            return_modularity (bool): Whether to return the modularity score.
            enforce_spatial (bool): Whether to apply spatial contiguity enforcement.
            score_metric (str): Metric to optimise during spatial enforcement.
            strategy (str): Strategy for spatial enforcement.
            verbose (bool): If True, prints enforcement diagnostics.
            measure_time (bool): If True, returns execution time.

        Returns:
            dict or tuple: Community assignments. Optionally returns modularity or execution time.
        """
        start_time = time.time() if measure_time else None

        kwargs = {'weights': 'weight'}
        if resolution is not None:
            kwargs['resolution'] = resolution

        partition = self.g_ig.community_multilevel(**kwargs)
        modularity = partition.modularity

        result = {self.g_ig.vs[i]["name"]: partition.membership[i]
                  for i in range(len(self.g_ig.vs))}

        if enforce_spatial:
            result, updated_score = self._apply_spatial_enforcement(result, modularity, score_metric,
                                                                    strategy, verbose)
            if measure_time:
                return result, updated_score, time.time() - start_time
            if return_modularity:
                return result, updated_score
            return result

        if measure_time:
            return result, modularity if return_modularity else None, time.time() - start_time
        if return_modularity:
            return result, modularity
        return result

    def run_leiden(self, resolution=None, return_modularity=False,
                   enforce_spatial=False, score_metric='modularity', strategy='min_impact_score',
                   verbose=False, measure_time=False):
        """Runs Leiden community detection.

        Args:
            resolution (float, optional): Resolution parameter for modularity optimisation.
            return_modularity (bool): Whether to return the modularity score.
            enforce_spatial (bool): Whether to apply spatial contiguity enforcement.
            score_metric (str): Metric to optimise during spatial enforcement.
            strategy (str): Strategy for spatial enforcement.
            verbose (bool): If True, prints enforcement diagnostics.
            measure_time (bool): If True, returns execution time.

        Returns:
            dict or tuple: Community assignments. Optionally returns modularity or execution time.
        """
        start_time = time.time() if measure_time else None

        kwargs = {
            'objective_function': "modularity",
            'weights': "weight",
            'n_iterations': self.trials
        }

        if resolution is not None:
            kwargs['resolution_parameter'] = resolution

        partition = self.g_ig.community_leiden(**kwargs)
        modularity = partition.modularity if return_modularity else None

        result = {self.g_ig.vs[i]["name"]: partition.membership[i]
                  for i in range(len(self.g_ig.vs))}

        if enforce_spatial:
            result, updated_score = self._apply_spatial_enforcement(result, modularity, score_metric,
                                                                    strategy, verbose)
            if measure_time:
                return result, updated_score, time.time() - start_time
            if return_modularity:
                return result, updated_score
            return result

        if measure_time:
            return result, modularity, time.time() - start_time
        if return_modularity:
            return result, modularity
        return result

    def run_infomap(self, preferred_modules=None, return_codelength=False,
                    enforce_spatial=False, score_metric='codelength',
                    strategy='min_impact_score', verbose=False,
                    measure_time=False):
        """Runs Infomap community detection.

        Args:
            preferred_modules (int, optional): Desired number of modules (used as hint).
            return_codelength (bool): Whether to return the final codelength.
            enforce_spatial (bool): Whether to apply spatial contiguity enforcement.
            score_metric (str): Metric to optimise during spatial enforcement.
            strategy (str): Strategy for spatial enforcement.
            verbose (bool): If True, prints enforcement diagnostics.
            measure_time (bool): If True, returns execution time.

        Returns:
            dict or tuple: Community assignments. Optionally returns codelength or execution time.
        """
        start_time = time.time() if measure_time else None

        flags = f"--two-level --silent --seed {self.seed}"
        if preferred_modules is not None:
            flags += f" --preferred-number-of-modules {preferred_modules}"

        im = Infomap(flags)
        im.num_trials = self.trials
        im.flow_model = "undirected"

        for edge in self.g_ig.es:
            im.add_link(edge.source, edge.target, edge["weight"])

        im.run()

        index_to_tractid = self.g_ig.vs["TractID"]
        community_dict = {index_to_tractid[node.node_id]: node.module_id for node in im.nodes}

        if enforce_spatial:
            community_dict, updated_score = self._apply_spatial_enforcement(
                community_dict, score=im.codelength, score_metric=score_metric,
                strategy=strategy, verbose=verbose
            )
            if measure_time:
                return community_dict, updated_score, time.time() - start_time
            if return_codelength:
                return community_dict, updated_score
            return community_dict

        if measure_time:
            return community_dict, im.codelength if return_codelength else None, time.time() - start_time
        if return_codelength:
            return community_dict, im.codelength
        return community_dict

    def _apply_spatial_enforcement(self, community_dict, score=None, score_metric='modularity',
                                   strategy='min_impact_score', verbose=False):
        """Applies spatial contiguity enforcement to community assignments.

        Args:
            community_dict (dict): Mapping of TractID to community label.
            score (float): 
            score_metric (str): Metric to optimise during enforcement.
            strategy (str): Strategy used to update communities.
            verbose (bool): If True, prints enforcement diagnostics.

        Returns:
            tuple: Updated community dictionary and final score.
        """
        if self.tracts is None:
            raise ValueError("GeoDataFrame with tracts is required for spatial enforcement but was not provided.")

        enforcer = SpatialContiguityEnforcer(
            gdf=self.tracts,
            community_dict=community_dict,
            score=score,
            g_ig=self.g_ig,
            verbose=verbose
        )
        updated_communities = enforcer.enforce_contiguity(score_metric=score_metric, strategy=strategy)
        updated_score = enforcer.get_final_score()

        if verbose:
            report = enforcer.get_contiguity_report()
            print("Spatial enforcement report:", report)
        return updated_communities, updated_score

    def compute_localisation_index(self, community_result, flow_matrix, weight_col="n_visits"):
        """Computes localisation index based on community assignment and a flow matrix.

        Args:
            community_result (dict): Mapping of TractID to community label.
            flow_matrix (DataFrame): Origin-destination matrix with flows between tracts.
            weight_col (str): Column in the flow matrix representing flow weights.

        Returns:
            DataFrame: Localisation index per community.
        """
        cm = CommunityMetrics(flow_matrix, self.tracts)
        li_df = cm.calculate_localisation_index(weigth_col=weight_col, community_assignment=community_result)
        return li_df