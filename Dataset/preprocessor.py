"""
preprocessing steps for the data, including normalization, binning, etc.
main part from 
scFormer/scformer/preprocess.py
"""

from typing import Dict, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np, anndata as ad, pandas as pd, scanpy as sc
from scipy.sparse import issparse,lil_matrix,csr_matrix
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
        logger.info("Initializing preprocessor ...")
        #logger.debug(f"dataset_para: {dataset_para}")
        self.para = dataset_para
        self.adata = None

    def load_adata(self, loc:str):
        """
        Load data from anndata object
        """
        logger.info(f"Load data from anndata object.")
        self.adata = sc.read_h5ad(loc)
   
    def save_adata(self, loc:str):
        """
        Save data to anndata object
        """
        logger.info(f"Save data to anndata object.")
        self.adata.write(loc)

    def from_csv(self,
                csv_loc:str,
                gene_name_mask:list,
                sample_id_idx:str,
                sample_mask:list,
                obs_name_mask:list,
                df:pd.DataFrame=None):
        """
        convert csv to anndata
        in:
            csv_loc: csv file location
            gene_name_mask: gene name mask if this colume in csv file should be included, then 
                           we set this gene name with True
            sample_id_idx: a index in a csv file that can use to sample items (normally sampel id)
            sample_mask: sample mask if this row in csv file should be included, then
                           we set this sample with True
            obs_name_mask: obs name mask if this colume in csv file should be included, then
                           we set this obs name with True
            df: (optional) a dataframe that can be used to convert to anndata
        out:
            anndata object
        """
        if df is None:
            logger.debug(f"Transfer a csv file to anndata object: {csv_loc}")
            df = pd.read_csv(csv_loc)
        else:
            logger.debug(f"Read raw data from a dataframe.")
            df = df
        logger.debug(f"csv file shape: {df.shape}")
        assert df.shape[1] == len(gene_name_mask) and df.shape[1] == len(obs_name_mask)
        assert df.shape[0] == len(sample_mask)

        # get gene name
        vocab = self.para.gene_vocab
        assert len(vocab)>0
        gene_list = df.columns[gene_name_mask].to_list()
        vocab_nb = len(vocab)
        # get samples needed
        new_df=df.iloc[sample_mask]
        logger.debug(f"new_df shape: {new_df.shape}")
        logger.debug(f"new_df keys: {new_df.keys()}")
        sample_nb = new_df.shape[0]
        # get obs for those samples
        obs = new_df.iloc[:,obs_name_mask]
        # get X
        counts = lil_matrix((sample_nb,vocab_nb),dtype=np.float32)
        for i in range(len(vocab)):
            if i % 2000 == 0: logger.debug(f"processing {i}/{len(vocab)}")
            if vocab[i] in gene_list:
                counts[:,i] = new_df[vocab[i]]

        counts = counts.tocsr()
        logger.debug(f"convert to anndata..")
        new = ad.AnnData(X=counts)
        new.var_names = vocab
        obs_names = pd.Index(new_df[sample_id_idx].to_list())

        new.obs = obs
        new.obs_names = obs_names
        #new.uns = {'log1p': {'base': 2}} #data.uns
        self.adata = new

    def run(self, batch_key: Optional[str] = None) -> Dict:
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
        is_logged = self.check_logged(self.adata, obs_key=key_to_process)

        # step 1: filter genes
        if self.para.filter_gene_by_counts:
            logger.info("Filtering genes by counts ...")
            sc.pp.filter_genes(
                self.adata,
                min_counts=self.para.filter_gene_by_counts
                if isinstance(self.para.filter_gene_by_counts, int)
                else None,
                inplace=True,
            )

        # step 2: filter cells
        if isinstance(self.para.filter_cell_by_counts, int):
            logger.info("Filtering cells by counts ...")
            sc.pp.filter_cells(
                self.adata,
                min_counts=self.para.filter_cell_by_counts
                if isinstance(self.para.filter_cell_by_counts, int)
                else None,
                inplace=True,
            )

        # step 3: normalize total,this part will automatically 
        # added into the adata object(inplace=False)
        if self.para.normalize_total:
            logger.info("Normalizing total counts ...")
            normed_ = sc.pp.normalize_total(
                self.adata,
                target_sum=self.para.normalize_total
                if isinstance(self.para.normalize_total, float)
                else None,
                layer=key_to_process,
                inplace=False,
            )["X"]
            key_to_process = self.para.result_normed_key \
                if self.para.result_normed_key is not None else key_to_process
            _set_obs_rep(self.adata, normed_, layer=key_to_process)

        # step 4: log1p
        # copy == False in default adata will be automatically changed
        if self.para.log1p:
            log1p_paras = {}
            logger.info("Log1p transforming ...")
            if is_logged:
                logger.warning(
                    "The input data seems to be already log1p transformed. "
                    "Set `log1p=False` to avoid double log1p transform."
                )
            if self.para.result_log1p_key:
                _set_obs_rep(
                    self.adata,
                    _get_obs_rep(self.adata, layer=key_to_process),
                    layer=self.para.result_log1p_key,
                )
                key_to_process = self.para.result_log1p_key
                log1p_paras["layer"] = key_to_process

            if self.para.log1p_base:
                log1p_paras["base"] = self.para.log1p_base
            sc.pp.log1p(self.adata, **log1p_paras)

        # step 5: subset hvg
        if self.para.subset_hvg:
            logger.info("Subsetting highly variable genes ...")
            if batch_key is None:
                logger.warning(
                    "No batch_key is provided, will use all cells for HVG selection."
                )
            sc.pp.highly_variable_genes(
                self.adata,
                layer=self.para.hvg_use_key,
                n_top_genes=self.para.subset_hvg
                if isinstance(self.para.subset_hvg, int)
                else None,
                batch_key=batch_key,
                flavor=self.para.hvg_flavor,
                subset=True,
                inplace=True, # this will automatically change the adata object
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
            layer_data = _get_obs_rep(self.adata, layer=key_to_process)
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
            self.adata.layers[self.para.result_binned_key] = np.stack(binned_rows)
            self.adata.obsm["bin_edges"] = np.stack(bin_edges)

        if self.para.preprocessed_loc is not None:
            logger.info(f"save preprocessed data to {self.para.preprocessed_loc}")
            self.adata.write(self.para.preprocessed_loc)
        logger.info("Preprocessing finished.")

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
 

    def to_data(self,data_type:str):
        """
        Get processed data from AnnData object
        Args:
            data_type: "X","normed","log1p","binned"
        Returns:
            processed data in sparse matrix format
        """
        data_type_list = ["X","normed","log1p","binned"]
        assert data_type in data_type_list, f"data_type must be in {data_type_list}"
        if data_type == "X":
            if self.para.result_normed_key is not None: 
                logger.warning(f"X is not normalised, check layer {self.para.result_normed_key}")
            if self.para.result_log1p_key is not None:
                logger.warning(f"X is not log1p transformed, check layer {self.para.result_log1p_key}")
            if self.para.result_binned_key is not None:
                logger.warning(f"X is not binned, check layer {self.para.result_binned_key}")
            return self.adata.X
        else:
            name_dict = {"normed":self.para.result_normed_key,
                        "log1p":self.para.result_log1p_key,
                        "binned":self.para.result_binned_key}
            data_type_name = name_dict[data_type]
            all_counts = (
                    self.adata.layers[data_type_name].A
                    if issparse(self.adata.layers[data_type_name])
                    else self.adata.layers[data_type_name])
            sparse_counts = csr_matrix(all_counts)
            return sparse_counts

    def to_label(self,
                label_key: str, 
                #----> for binarize label
                binarize:bool=False,
                bins:np.ndarray=None,
                bin_nb: int=None,bin_min:float=None,bin_max:float=None,
                save_in_obs:bool=True,
                
                ) -> None:
        """
        get label from adata.obs[label_key]
        if binarize is True, then we will binarize the label
        if save_in_obs is True, then we will save the label in adata.obs[label_key]
        Args:
            label_key (:class:`str`):
                The key of :class:`AnnData.obs` to use as label
            binarize (:class:`bool`)(optional):
                If True, we will binarize the label
            bins (:class:`np.ndarray`)(optional):
                The bins to binarize the label
            bin_nb (:class:`int`)(optional):
                The number of bins to binarize the label
            bin_min (:class:`float`)(optional):
                The min value of bins to binarize the label
            bin_max (:class:`float`)(optional):
                The max value of bins to binarize the label
            save_in_obs (:class:`bool`)(optional):
                If True, we will save the label in adata.obs[label_key]
        Returns:
            label (:class:`torch.tensor`):
                The label of the data
            class_weight (:class:`torch.tensor`):
                The class weight of the each category
        """
        logger.info(f"Binarize label {label_key} in obs_names")
        original_label = self.adata.obs[label_key] 
        if binarize:
            if bins is None:
                assert bin_nb is not None 
                if bin_min is None: bin_min = original_label.min()
                if bin_max is None: bin_max = original_label.max()
                bins = np.linspace(bin_min, bin_max, bin_nb)
            bin_names = np.arange(bin_nb)
            digitized = np.digitize(original_label, bins)
            binned_label = bin_names[digitized-1]
            if save_in_obs:
                obs_name = f"{label_key}_binned"
                self.adata.obs[obs_name] = binned_label
            np_label = binned_label
        else:
            np_label = original_label.to_numpy()

        class_num = np.unique(np_label, return_counts=True)[1].tolist()
        class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
        label = torch.from_numpy(np_label).unsqueeze(1)
        return label,class_weight
    
    def to_dataloader(self,dataset:Dataset):
        """
        Get dataloader from dataset
        Args:
            dataset: dataset to get dataloader
        Returns:
            dataloader
        """
        dataloader = DataLoader(dataset, 
                                batch_size=self.para.batch_size, 
                                shuffle=self.para.shuffle, 
                                num_workers=self.para.num_workers,
                                **self.para.additional_dataloader_para,
                                )
        return dataloader


        