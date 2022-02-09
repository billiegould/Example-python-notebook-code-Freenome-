from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Any

from pineappleflow.core.logging.logger import pprint_logger
from pineappleflow.core.artifacts.matrix import Matrix
from pineappleflow.core.components.transformer import Transformer
from pineapple.contrib.specs.transformer_specs import WeightByRegionStatisticSpec


class WeightByRegionStatistic(Transformer):
    __spec_cls__ = WeightByRegionStatisticSpec

    @classmethod
    def from_spec(cls, spec: WeightByRegionStatisticSpec) -> WeightByRegionStatistic:
        return cls(**spec["kwargs"])

    def __init__(
        self,
        path_to_reg_stat_file: str,
        stat_col_name: str,
        cmeta_gen_loc_cols: list,
        top_pct_reg_to_retain: Any[None, float],
    ):
        """The transformer re-weights methyl regions according to their statistical value as annotated in an external
        file (path_to_reg_stat_file). The file must have a regions column (chrm:start-end) and a statistic column
        specified by stat_col_name. If top_pct_reg_to_retain is specified, all regions above the specified percentile
        will be retained (weighted as 1) and all below the cutoff will be discarded (weighted as 0). If the param is
        not specified (default=None) the regions are weighted using the normalized statistic in the input file.
               Args:
                   path_to_reg_stat_file ([str]): gcs full path to file containing region annotations in column
                       "region" (formatted as chr:start-stop) and stat values in column stat_col_name specified below
                   stat_col_name: header of column in stat input file containing the region cutoff metric
                   top_pct_reg_to_retain ([float]): regions above this percentile will be weighted 1 and below will
                       be weighted 0. Default=None, where the normalized stat value itself will be used as the weight.
                   cmeta_gen_loc_cols ([str]): list of length 3 indicating the names of the keys in the matrix column
                       metadata object that specify chrom, start and end of the panel regions
               Example Args:
                   path_to_reg_stat_file = "gs://reference-datafiles/freenome_panels/cpg_dense_research_v2/
                       multi_cancer_region_subsets/VP2_chmfc_reg_rate_diffs.csv"
                   stat_col_name = "med_hmf_rate_diff"
                   top_pct_reg_to_retain = 50.0
                   cmeta_gen_loc_cols = ["seqname", "start", "end"]
        """
        super().__init__()
        self.path_to_reg_stat_file = path_to_reg_stat_file
        self.stat_col_name = stat_col_name
        self.top_pct_reg_to_retain = top_pct_reg_to_retain
        self.cmeta_gen_loc_cols = cmeta_gen_loc_cols

    def weighting_vector(
        self,
        cmeta: np.array,
        path_to_reg_stat_file: str,
        stat_col_name: str,
        top_pct_reg_to_retain: Any[None, float],
        cmeta_gen_loc_cols: List[str] = ["seqname", "start", "end"],
    ) -> np.array:
        """
        Computes weighting vector: weight = normalized statistic from the infile if top_pct_reg_to_Retain==None.
        Else weight==1.0 when region statistic > top_pct_reg_to_retain, else 0. i.e. retains only regions above the
        percentile specified.
        All regions in the model must be present in the region weights input file.
        """
        df_reg_stats = pd.read_csv(filepath_or_buffer=path_to_reg_stat_file, header=0)

        # make sure we have all the right column names.
        bed_cols_not_present_bool = [col not in cmeta[0].keys() for col in cmeta_gen_loc_cols]
        assert not any(
            bed_cols_not_present_bool
        ), f"Spec. file keys missing from model column metadata: {np.array(cmeta_gen_loc_cols)[bed_cols_not_present_bool]}"
        assert "regions" in df_reg_stats.columns, "region weights file is missing a 'regions' column"
        assert stat_col_name in df_reg_stats.columns, f"region weights file is missing column {stat_col_name}"

        matrix_regions = [
            f"{d[cmeta_gen_loc_cols[0]]}:{d[cmeta_gen_loc_cols[1]]}-{d[cmeta_gen_loc_cols[2]]}" for d in cmeta
        ]
        reg_in_stat_file = [reg for reg in matrix_regions if reg in df_reg_stats["regions"].values]
        assert all(
            [reg in df_reg_stats["regions"].values for reg in matrix_regions]
        ), f"Error: Only {len(reg_in_stat_file)} of {len(matrix_regions)} model regions are in the region weights file."

        stats_dict = dict(zip(df_reg_stats["regions"], df_reg_stats[stat_col_name]))
        _max = max(stats_dict.values())
        pctls = []
        if top_pct_reg_to_retain is not None:
            assert (
                type(top_pct_reg_to_retain) == float
            ), f"Error: top_pct_reg_to_retain ({top_pct_reg_to_retain}) must be float or NULL"
            stat_vals = [stats_dict[reg] for reg in reg_in_stat_file]
            for reg in matrix_regions:
                rank = list(np.array(reg_in_stat_file)[np.argsort(stat_vals)]).index(reg)
                reg_pctile = float(rank) / len(reg_in_stat_file) * 100.0
                pctls.append(reg_pctile)
                # retain only the top percentage of regions
                if reg_pctile >= 100 - top_pct_reg_to_retain:
                    stats_dict.update({reg: _max})
                else:
                    stats_dict.update({reg: 0.0})
        # normalize the region statistics to weights
        wv = np.array([stats_dict[reg] for reg in matrix_regions]) / _max

        assert not all([w == 0 for w in wv]), f"Error: All region weights equal 0.\n{dict(zip(pctls,wv))}"

        return wv

    def fit(self, matrix: Matrix) -> WeightByRegionStatistic:
        # use fit to compute weight vector only using training data
        wv = self.weighting_vector(
            path_to_reg_stat_file=self.path_to_reg_stat_file,
            stat_col_name=self.stat_col_name,
            top_pct_reg_to_retain=self.top_pct_reg_to_retain,
            cmeta=matrix.column_metadata,
            cmeta_gen_loc_cols=self.cmeta_gen_loc_cols,
        )
        self.wv = wv

        pprint_logger(
            name="WeightByRegionStatistic",
            msg_list=[
                f"Using file {self.path_to_reg_stat_file} to weight methyl regions",
                f"Using region statistic to retain {self.top_pct_reg_to_retain} pct of methyl regions",
            ],
            level="warn",
        )
        return self

    def transform(self, matrix: Matrix) -> Matrix:

        # multiply the methylated fragment counts by the weight vector for every sample
        nsamp = matrix.x.shape[0]
        nreg = matrix.x.shape[1]
        hmfc_plane = np.transpose(np.repeat(self.wv[..., np.newaxis], nsamp, axis=1))
        total_frags_plane = np.ones((nsamp, nreg))
        wmatrix = np.multiply(matrix.x, np.stack([hmfc_plane, total_frags_plane], axis=2))

        # update column metadata with weight.
        cmeta_wweight = matrix.column_metadata
        for idx in range(len(cmeta_wweight)):
            cmeta_wweight[idx]["weight"] = self.wv[idx]

        return matrix.replace_x_and_col_metadata(x=wmatrix, column_metadata=cmeta_wweight)
