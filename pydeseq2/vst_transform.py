# pylint: disable=invalid-name
"""PCA transformation."""
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from meow import types
from meow.dataset.multidataset import MultiDatasetSplit
from meow.transforms.base_transform import BaseTransform
from scipy.stats import trim_mean

from pydeseq2.dds import DeseqDataSet

BP_LENGTH = "/home/owkin/project/dataset/new_tcga_data/recount3_gene_bp_length.parquet"


class VSTTransform(BaseTransform):
    """DESeq2' Variance Stabilizing Transform.

    Parameters
    ----------
    use_design: bool, default=False
        Whether to use design factors for dispersion estimation.
    min_disp: float, default=1e-8
        Minimum dispersion value to use for dispersion estimation.
    fit_type: str, default="mean"
        Type of dispersion estimation to use. Either "mean" or "parametric".
    correct_for_gene_lengths: bool, default=True
        Whether to correct for gene lengths. Uses the transcript gene lengths from
        recount3.
    design_factors: str, default="origin"
        Design factors to use for dispersion estimation.

    Attributes
    ----------
    use_design: bool
        Whether to use design factors for dispersion estimation.
    min_disp: float
        Minimum dispersion value to use for dispersion estimation.
    fit_type: str
        Type of dispersion estimation to use. Either "mean" or "parametric".
    correct_for_gene_lengths: bool
        Whether to correct for gene lengths.
    design_factors: str
        Design factors to use for dispersion estimation.
    metadata_columns: pd.Index
        Metadata columns (to be used during transform).
    dds: DeseqDataSet
        Deseq2 dataset (fitted during train).
    """

    def __init__(
        self,
        use_design: bool = False,
        min_disp: float = 1e-8,
        fit_type: Literal["parametric", "mean"] = "mean",
        correct_for_gene_lengths: bool = True,
        design_factors: str = "origin",
    ):
        self.use_design = use_design
        self.min_disp = min_disp
        self.fit_type = fit_type
        self.design_factors = design_factors
        self.correct_for_gene_lengths = correct_for_gene_lengths
        if correct_for_gene_lengths:
            self.gene_lengths: pd.Series = pd.read_parquet(BP_LENGTH)["bp_length"]
            self.reference_gene_lengths = self.gene_lengths.copy()
        self.metadata_columns: pd.Index
        self.dds: DeseqDataSet

    def fit_array(self, X: pd.DataFrame, metadata: pd.DataFrame):
        """Fit genewise dispersion and dispersion trend using train data."""
        if (X % 1 != 0).any().any():
            logger.warning("X is not integer, will be rounded to int.")
            X = X.round()

        if self.correct_for_gene_lengths:
            # Assert that X is a dataframe and that the columns are either gene names or
            # gene ids
            assert isinstance(X, pd.DataFrame) and (
                X.columns.str.startswith("ENSG").any()
                or not X.columns.intersection(
                    self.gene_lengths.index.get_level_values("gene_name")
                )
            ), """X should be a dataframe with either gene names or gene ids as columns
               with `self.correct_for_gene_lengths` set to `True`."""
            # Drop either gene names or gene ids based on X columns
            if X.columns.str.startswith("ENSG").sum() > (X.shape[1] / 2):
                gene_lengths = self.reference_gene_lengths.reset_index().set_index(
                    "gene_id"
                )["bp_length"]
                # Remove ensg version
                gene_lengths.index = gene_lengths.index.str.split(".").str[0]
                # Take mean length per ensg (should be the same for all versions)
                gene_lengths = gene_lengths.groupby(gene_lengths.index).mean()
                self.gene_lengths = gene_lengths.loc[
                    gene_lengths.index.intersection(X.columns)
                ]
            else:
                gene_lengths = self.reference_gene_lengths.reset_index().set_index(
                    "gene_name"
                )["bp_length"]
                self.gene_lengths = gene_lengths.loc[
                    gene_lengths.index.intersection(X.columns)
                ]
            assert (
                self.gene_lengths.shape[0] == X.shape[1]
            ), """Gene lengths should have
                the same number of genes as the number of columns in X."""

        self.dds = DeseqDataSet(
            counts=X,
            metadata=metadata,
            design_factors=self.design_factors,
            refit_cooks=True,
            n_cpus=None,
        )
        self.dds.fit_size_factors()
        self.dds.fit_genewise_dispersions()
        self.dds.fit_dispersion_trend()

        # Save gene lengh correction coefficients
        if self.correct_for_gene_lengths:
            vst = self.compute_vst(self.dds.layers["normed_counts"])
            # Regress out the effects of gene lengths on the gene-wise average vst.
            Y = np.log2(self.gene_lengths).values
            Y_centered = Y - Y.mean()
            Y_centered = Y_centered.reshape(-1, 1)
            w = np.linalg.lstsq(Y_centered, vst.mean(axis=0), rcond=None)[0]
            self.dds.uns["gene_length_correction"] = w
            self.dds.uns["centered_gene_lengths"] = Y_centered

    def transform_array(self, X: pd.DataFrame, metadata: pd.DataFrame):
        """Transform data using fitted dispersion parameters."""
        if (X % 1 != 0).any().any():
            logger.warning("X is not integer, will be rounded to int.")
            X = X.round()
        _dds = DeseqDataSet(
            counts=X,
            metadata=metadata,
            design_factors=self.design_factors,
            refit_cooks=True,
            n_cpus=None,
        )
        _dds.fit_size_factors()
        normed_counts = _dds.layers["normed_counts"]

        vst = self.compute_vst(normed_counts)

        if self.correct_for_gene_lengths:
            Y_centered = self.dds.uns["centered_gene_lengths"]
            w = self.dds.uns["gene_length_correction"]
            vst = vst - Y_centered @ w.T

        return vst

    def fit(self, train_dataset: MultiDatasetSplit):  # pylint: disable=unused-argument
        """Fit VST on meow multidataset."""
        self.metadata_columns = train_dataset.features_dataframes["metadata"].columns
        self.fit_array(
            train_dataset.get_x("rnaseq", True),
            train_dataset.features_dataframes["metadata"],
        )
        return self

    def transform(self, batch: types.Batch):
        """Perform VST transform on meow Batch."""
        if "rnaseq" in batch["features"]:
            X = batch["features"]["rnaseq"]
            metadata = batch["features"]["metadata"]
            metadata = pd.DataFrame(metadata, columns=self.metadata_columns)

            vst = self.transform_array(X, metadata)
            batch["features"]["rnaseq_aligned"] = vst
            batch["feature_names"]["rnaseq_aligned"] = batch["feature_names"]["rnaseq"]

        return batch

    def compute_vst(self, normed_counts: np.ndarray):
        """Compute VST from normed counts."""
        if self.fit_type == "parametric":
            a0, a1 = self.dds.uns["trend_coeffs"]
            cts = normed_counts
            vst = np.log2(
                (1 + a1 + 2 * a0 * cts + 2 * np.sqrt(a0 * cts * (1 + a1 + a0 * cts)))
                / (4 * a0)
            )
        elif self.fit_type == "mean":
            gene_dispersions = self.dds.varm["genewise_dispersions"]
            use_for_mean = gene_dispersions > 10 * self.min_disp
            mean_disp = trim_mean(gene_dispersions[use_for_mean], proportiontocut=0.001)
            vst = (
                2 * np.arcsinh(np.sqrt(mean_disp * normed_counts))
                - np.log(mean_disp)
                - np.log(4)
            ) / np.log(2)
        else:
            raise ValueError(f"Unknown fit_type: {self.fit_type}")

        return vst
