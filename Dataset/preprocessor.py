"""
preprocessing steps for the data, including normalization, binning, etc.
main part from 
scFormer/scformer/preprocess.py
"""

from typing import Dict, Optional, Union

import numpy as np, anndata as ad, pandas as pd, scanpy as sc
from scipy.sparse import issparse,lil_matrix
from scanpy.get import _get_obs_rep, _set_obs_rep
from anndata import AnnData

from scLLM.Dataset.paras import Dataset_para
from scLLM import logger

class Preprocessor:
    """
    Prepare data into training, valid and test split. Normalize raw expression
    values, binning or using other transform into the preset model input format.
    """

    def __init__(
        self,
        dataset_para: Dataset_para,
    ):
        r"""
        Set up the preprocessor, use the args to config the workflow steps.
        """
        self.para = dataset_para

    def __call__(self, adata: AnnData, batch_key: Optional[str] = None) -> Dict:
        """
        format controls the different input value wrapping, including categorical
        binned style, fixed-sum normalized counts, log1p fixed-sum normalized counts, etc.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        batch_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        key_to_process = self.para.use_key
        # preliminary checks, will use later
        if key_to_process == "X":
            key_to_process = None  # the following scanpy apis use arg None to use X
        is_logged = self.check_logged(adata, obs_key=key_to_process)

        # step 1: filter genes
        if self.para.filter_gene_by_counts:
            logger.info("Filtering genes by counts ...")
            sc.pp.filter_genes(
                adata,
                min_counts=self.para.filter_gene_by_counts
                if isinstance(self.para.filter_gene_by_counts, int)
                else None,
            )

        # step 2: filter cells
        if isinstance(self.para.filter_cell_by_counts, int):
            logger.info("Filtering cells by counts ...")
            sc.pp.filter_cells(
                adata,
                min_counts=self.para.filter_cell_by_counts
                if isinstance(self.para.filter_cell_by_counts, int)
                else None,
            )

        # step 3: normalize total
        if self.para.normalize_total:
            logger.info("Normalizing total counts ...")
            normed_ = sc.pp.normalize_total(
                adata,
                target_sum=self.para.normalize_total
                if isinstance(self.para.normalize_total, float)
                else None,
                layer=key_to_process,
                inplace=False,
            )["X"]
            key_to_process = self.para.result_normed_key or key_to_process
            _set_obs_rep(adata, normed_, layer=key_to_process)

        # step 4: log1p
        if self.para.log1p:
            logger.info("Log1p transforming ...")
            if is_logged:
                logger.warning(
                    "The input data seems to be already log1p transformed. "
                    "Set `log1p=False` to avoid double log1p transform."
                )
            if self.para.result_log1p_key:
                _set_obs_rep(
                    adata,
                    _get_obs_rep(adata, layer=key_to_process),
                    layer=self.para.result_log1p_key,
                )
                key_to_process = self.para.result_log1p_key
            sc.pp.log1p(adata, layer=key_to_process)

        # step 5: subset hvg
        if self.para.subset_hvg:
            logger.info("Subsetting highly variable genes ...")
            if batch_key is None:
                logger.warning(
                    "No batch_key is provided, will use all cells for HVG selection."
                )
            sc.pp.highly_variable_genes(
                adata,
                layer=self.para.hvg_use_key,
                n_top_genes=self.para.subset_hvg
                if isinstance(self.para.subset_hvg, int)
                else None,
                batch_key=batch_key,
                flavor=self.para.hvg_flavor,
                subset=True,
            )

        # step 6: binning
        if self.para.binning:
            logger.info("Binning data ...")
            if not isinstance(self.para.binning, int):
                raise ValueError(
                    "Binning arg must be an integer, but got {}.".format(self.para.binning)
                )
            n_bins = self.para.binning  # NOTE: the first bin is always a spectial for zero
            binned_rows = []
            bin_edges = []
            layer_data = _get_obs_rep(adata, layer=key_to_process)
            layer_data = layer_data.A if issparse(layer_data) else layer_data
            for row in layer_data:
                non_zero_ids = row.nonzero()
                non_zero_row = row[non_zero_ids]
                bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
                # bins = np.sort(np.unique(bins))
                # NOTE: comment this line for now, since this will make the each category
                # has different relative meaning across datasets
                non_zero_digits = np.digitize(non_zero_row, bins)
                assert non_zero_digits.min() >= 1
                assert non_zero_digits.max() <= n_bins - 1
                binned_row = np.zeros_like(row, dtype=np.int64)
                binned_row[non_zero_ids] = non_zero_digits
                binned_rows.append(binned_row)
                bin_edges.append(np.concatenate([[0], bins]))
            adata.layers[self.para.result_binned_key] = np.stack(binned_rows)
            adata.obsm["bin_edges"] = np.stack(bin_edges)

    def check_logged(self, adata: AnnData, obs_key: Optional[str] = None) -> bool:
        """
        Check if the data is already log1p transformed.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        obs_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        data = _get_obs_rep(adata, layer=obs_key)
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False

        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False

        return True
    
    def csv_to_anndata(self,
                        csv_loc:str,
                        gene_name_mask:list,
                        sample_id_idx:int,
                        sample_mask:list,
                        obs_name_mask:list):
        """
        convert csv to anndata
        in:
            csv_loc: csv file location
            gene_name_mask: gene name mask if this colume in csv file is gene name with True
            sample_id_idx: sample id mask if a colume in csv file is sample id with True (only one True in a row)
            sample_mask: sample mask if this row in csv file is sample with True
            obs_name_mask: obs name mask if this colume in csv file is obs name with True
        out:
            anndata object
        """
        logger.debug(f"Transfer a csv file to anndata object: {csv_loc}")
        df = pd.read_csv(csv_loc)
        logger.debug(f"csv file shape: {df.shape}")
        assert df.shape[1] == len(gene_name_mask) and df.shape[1] == len(obs_name_mask)
        assert df.shape[0] == len(sample_mask)

        # get gene name
        assert len(vocab)>0
        vocab = self.para.gene_vocab
        gene_list = df.columns[gene_name_mask].to_list()
        gene_nb = len(gene_list)
        # get samples needed
        new_df=df.iloc[sample_mask]
        sample_nb = new_df.shape[0]
        # get obs for those samples
        obs = new_df.iloc[:,obs_name_mask]
        # get X
        counts = lil_matrix((sample_nb,gene_nb),dtype=np.float32)
        for i in range(len(vocab)):
            if i % 2000 == 0: print(f"processing {i}/{len(vocab)}")
            if vocab[i] in gene_list:
                counts[:,i] = new_df[vocab[i]]

        counts = counts.tocsr()
        logger.debug(f"convert to anndata..")
        new = ad.AnnData(X=counts)
        new.var_names = vocab
        new.obs_names = new_df[new_df.columns[sample_id_idx]].to_list()
        new.obs = obs
        new.uns = {'log1p': {'base': 2}} #data.uns
        return new


