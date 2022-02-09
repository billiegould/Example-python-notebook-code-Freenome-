from __future__ import annotations

from typing import Tuple

import numpy as np
from pineappleflow.core.artifacts.matrix import Matrix
from pineappleflow.core.components.transformer import Transformer
from pineappleflow.core.logging.logger import pprint_logger

from pineapple.contrib.specs.transformer_specs import TissueThresholdTransformerSpec


def get_population_frequency_per_region(
    tissue_per_region_rates: np.ndarray,
    region_names: np.ndarray,
    region_name_for_defining_what_is_on: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        tissue_per_region_rates: this should be an n_sample x n_region dimensional array of rates of some form.
        region_names: a list of names per region, should be an n_region length array of strings.
        region_name_for_defining_what_is_on: the string name of the region you want to use to define what "on"
            looks like in each sample.
    Returns:
        ( cutoff used per sample to determine which regions are "on"
            tissue_on_regions: the boolean n_sa
            tissue_sample_on_cutoff: the floatmple x n_region matrix of which regions pass the "on" cutoff per sample
            pop_freq_per_region: the fraction of samples that have each region on, length is n_region.
        )
    """

    assert (
        len(tissue_per_region_rates.shape) == 2
    ), f"Unexpected shape {tissue_per_region_rates.shape} should be length 2"
    if region_name_for_defining_what_is_on not in set(region_names):
        raise ValueError(
            f"Error, did not find {region_name_for_defining_what_is_on} in region names. "
            f"First 5 regions: {region_names[:5]}. "
        )
    # select a region to define what 'on' looks like, the region may have several segments
    region_selector_for_on = region_names == region_name_for_defining_what_is_on
    sample_on_cutoffs_lst = []
    # for each sample, calculate the mean for the selected region(s) and append the sample cutoff for the region?
    for i in range(len(tissue_per_region_rates)):
        sample_hmfs = tissue_per_region_rates[i]
        mean_on = np.mean(sample_hmfs[region_selector_for_on])
        on_regions = (sample_hmfs >= (mean_on * 0.75)) & (sample_hmfs <= mean_on * 1.5)
        assert np.sum(on_regions) > 1, "need more than one on region per sample"
        sample_cutoff = max(
            mean_on - 1.960 * np.std(sample_hmfs[on_regions], ddof=1),
            0.1,
        )
        sample_on_cutoffs_lst.append(sample_cutoff)

    # one cutoff value per tissue sample
    tissue_sample_on_cutoffs: np.ndarray = np.array(sample_on_cutoffs_lst, dtype=np.float)
    # determine which tissue regions are on in by comparing to the sample cutoff for the chosen region
    tissue_on_regions = tissue_per_region_rates >= tissue_sample_on_cutoffs[:, np.newaxis]
    # average the true false on staus of tissue regions to get an average pop freq per region
    pop_freq_per_region = np.mean(tissue_on_regions, axis=0)
    assert len(pop_freq_per_region) == tissue_per_region_rates.shape[1]
    return tissue_sample_on_cutoffs, tissue_on_regions, pop_freq_per_region


class TissueThresholdTransformer(Transformer):
    __spec_cls__ = TissueThresholdTransformerSpec

    @classmethod
    def from_spec(cls, spec: TissueThresholdTransformerSpec) -> TissueThresholdTransformer:
        return cls(**spec["kwargs"])

    def __init__(
        self,
        aux_id_for_population_frequency_tissue_data: str,
        minimal_population_frequency_for_region_inclusion: float,
        region_name_for_defining_what_is_on: str,
        column_metadata_field_for_region_name: str,
    ):
        """
        Args:

            aux_id_for_population_frequency_tissue_data: tissue data aux id. Tissue samples should be queried with
                this aux id, they should all have high tissue dna fraction, so late stage plasma samples with
                confirmed signal would also work here in place of tissue.
            minimal_population_frequency_for_region_inclusion: the minimum fraction of samples with each region "on"
                to consider that region of sufficiently high population frequency in a set of high ctDNA samples.
            region_name_for_defining_what_is_on: A region known to be constitutively active, to use to determine what
                "on" looks like in each sample in the tissue aux id.
            column_metadata_field_for_region_name: Which column metadata field to use to look up region names.
        Returns:
            Fitted self.
        """

        self.aux_id_for_population_frequency_tissue_data = aux_id_for_population_frequency_tissue_data
        self.region_name_for_defining_what_is_on = region_name_for_defining_what_is_on
        self.column_metadata_field_for_region_name = column_metadata_field_for_region_name
        self.minimal_population_frequency_for_region_inclusion = minimal_population_frequency_for_region_inclusion
        self.features_to_keep: np.ndarray = None
        self.sample_on_cutoffs: np.ndarray = None
        self.tissue_on_regions: np.ndarray = None
        self.pop_freq_per_region: np.ndarray = None

    def fit(self, matrix: Matrix) -> TissueThresholdTransformer:
        """
        Given a set of tissue samples methylation data, calculate methylation cuttoff thresholds for each region
        to be used to transform sample data and eliminate regions.

        Args:
            matrix: matrix of methylation values for tissue samples. samples x regions x methreads x unmethreads
        """

        # 1. get x for the tissue samples that are sourced from this aux field
        # Select the tissue samples to use from matrix of tissue samples
        tissue_mask = [rm.source == self.aux_id_for_population_frequency_tissue_data for rm in matrix.row_metadata]
        if np.sum(tissue_mask) == 0:
            raise ValueError("no tissue samples with specified name found")
        # samples x regions x 2
        # filter the dataset including the sample dimension
        tissue_data = matrix.x[tissue_mask]

        # add a pseudo count of 1 to n methyl and 1 to n unmethyl, or 1 to numerator and 2 to denominator
        # samples x regions
        if tissue_data.shape[-1] != 2 or len(tissue_data.shape) != 3:
            raise ValueError(f"expected last dimension length 2 but found {tissue_data.shape}")
        tissue_hmf_rates = (tissue_data[..., 0].astype(np.float) + 1.0) / (tissue_data[..., 1].astype(np.float) + 2.0)

        # get the region names
        region_names = np.array([c.get(self.column_metadata_field_for_region_name) for c in matrix.column_metadata])
        tissue_sample_on_cutoffs, tissue_on_regions, pop_freq_per_region = get_population_frequency_per_region(
            tissue_per_region_rates=tissue_hmf_rates,
            region_names=region_names,
            region_name_for_defining_what_is_on=self.region_name_for_defining_what_is_on,
        )
        self.sample_on_cutoffs = tissue_sample_on_cutoffs
        self.tissue_on_regions = tissue_on_regions
        self.pop_freq_per_region = pop_freq_per_region
        self.features_to_keep = self.pop_freq_per_region >= self.minimal_population_frequency_for_region_inclusion
        pprint_logger(
            name="TissueThresholdTransformer",
            msg_list=[
                f"Found {len(tissue_data)} samples to estimate population frequency from using "
                f"{self.region_name_for_defining_what_is_on} as a reference gene region. "
                f"From these there were {np.sum(self.features_to_keep)} regions active in "
                f"over {self.minimal_population_frequency_for_region_inclusion} population frequency."
            ],
            level="warn",
        )

        return self

    def transform(self, matrix: Matrix) -> Matrix:
        """filters the sample data down to the features to keep based on tissue methylation cutoffs.
        Args:
            matrix: data to transform
        Returns:
            x (numpy array): transformed data
        """
        return matrix[:, self.features_to_keep]
