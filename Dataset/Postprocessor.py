import torch
from typing import Dict, Optional, Union,Tuple,List
import numpy as np, scanpy as sc, anndata as ad,pandas as pd

from scipy.sparse import issparse,lil_matrix,csr_matrix
from scanpy.get import _get_obs_rep, _set_obs_rep
from anndata import AnnData
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import AutoTokenizer, BertTokenizer
from torch.utils.data import DataLoader, Dataset

from scLLM import logger
from scLLM.Dataset.paras import Dataset_para
from scLLM.Dataset.Vocab import GeneVocab


class scBERTPostprocessor:
    def __init__(self,paras:Dataset_para,vocab:GeneVocab) -> None:
        self.paras = paras
        self.my_vocab = vocab

    def run(self,
                 adata,
                ):
        # data part
        data_type = self.paras.data_layer_name
        # label part
        label_key = self.paras.label_key
        cls_nb = self.paras.cls_nb
        binarize = self.paras.binarize # method to binarize label
        bins = self.paras.bins
        bin_min = self.paras.bin_min
        bin_max = self.paras.bin_max
        save_in_obs = self.paras.save_in_obs

        # split part
        n_splits = self.paras.n_splits
        test_size = self.paras.test_size
        random_state = self.paras.random_state

        # get all data
        all_data = self.to_data(adata=adata,data_type=data_type)
        # get all label
        all_label,class_weight = self.to_label(adata=adata,
                                  label_key=label_key,
                                  #----> for binarize label
                                    binarize=binarize,
                                    bins=bins,
                                    bin_nb=cls_nb,bin_min=bin_min,bin_max=bin_max,
                                    save_in_obs=save_in_obs,
                                  )
        # split train test
        D_train,D_val = self.split_train_test(all_data,all_label,
                                              # for how to split
                                              n_splits=n_splits, 
                                              test_size=test_size, 
                                              random_state=random_state,
                                              )
        # for train part
        trainset = self.create_dataset(D_train,
                                       cls_nb=cls_nb,
                                       )
        # for val part
        valset = self.create_dataset(D_val,
                                        cls_nb=cls_nb,
                                        )
        return trainset,valset,class_weight
    ##############################################################################################################
    #  tokenize steps for scBERT
    #############################################################################################################

    def to_data(self,adata,data_type:str):
        """
        Get processed data from AnnData object
        Args:
            data_type: "X","normed","log1p","binned"
        Returns:
            processed data in sparse matrix format
        """
        data_type_list = ["X","X_normed","X_log1p","X_binned"]
        assert data_type in data_type_list, f"data_type must be in {data_type_list}"
        if data_type == "X":
            if self.paras.result_normed_key is not None: 
                logger.warning(f"X is not normalised, check layer {self.paras.result_normed_key}")
            if self.paras.result_log1p_key is not None:
                logger.warning(f"X is not log1p transformed, check layer {self.paras.result_log1p_key}")
            if self.paras.result_binned_key is not None:
                logger.warning(f"X is not binned, check layer {self.paras.result_binned_key}")
            return adata.X
        else:
            name_dict = {"X_normed":self.paras.result_normed_key,
                        "X_log1p":self.paras.result_log1p_key,
                        "X_binned":self.paras.result_binned_key}
            data_type_name = name_dict[data_type]
            all_counts = (
                    adata.layers[data_type_name].A
                    if issparse(adata.layers[data_type_name])
                    else adata.layers[data_type_name])
            sparse_counts = csr_matrix(all_counts)
            return sparse_counts
        
    def to_label(self,
                adata:AnnData,
                label_key: str, 
                #----> for binarize label
                binarize:str=None,
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
            binarize (:class:`str`)(optional): ["quantile",""]
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
        logger.info(f"Discritize label {label_key} in obs_names")
        original_label = adata.obs[label_key] 
        if binarize is not None:
            assert binarize in ["equal_width","equal_instance"]
            if bins is None:
                assert bin_nb is not None 
                if bin_min is None: bin_min = original_label.min()
                if bin_max is None: bin_max = original_label.max()
                if binarize == "equal_width":
                    bins = np.linspace(bin_min, bin_max, bin_nb)
                elif binarize == "equal_instance":
                    c_label = np.sort(original_label.to_numpy().flatten())
                    bins = np.array([ c_label[int(((len(c_label)-1)/bin_nb)*i)] for i in range(bin_nb)])
            bin_names = np.arange(bin_nb)
            digitized = np.digitize(original_label, bins)
            binned_label = bin_names[digitized-1]
            if save_in_obs:
                obs_name = f"{label_key}_binned"
                adata.obs[obs_name] = binned_label
            np_label = binned_label
        else:
            np_label = original_label.to_numpy()

        class_num = np.unique(np_label, return_counts=True)[1].tolist()
        class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
        label = torch.from_numpy(np_label).unsqueeze(1)
        return label,class_weight

    def split_train_test(self,all_data,all_label,
                          n_splits=1, test_size=0.2, random_state=2023):
        from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
        #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
        sss = StratifiedShuffleSplit(n_splits=n_splits, 
                                     test_size=test_size, 
                                     random_state=random_state)

        idx_tr,idx_val = next(iter(sss.split(all_data, all_label)))
        data_train, label_train = all_data[idx_tr], all_label[idx_tr]
        data_val, label_val = all_data[idx_val], all_label[idx_val]
        return [data_train,label_train],[data_val,label_val]
    
    def create_dataset(self,data_and_label:list,cls_nb:int=5):
        [data,label] = data_and_label
        from scLLM.Dataset.dataset import SCDataset
        dataset = SCDataset(data, label,cls_nb=cls_nb)
        return dataset


###################################################################
#     postprocessor for scGPT include tokenizer and mask
###################################################################
class scGPTPostprocessor:
    def __init__(self,paras:Dataset_para,vocab:GeneVocab):
        self.para = paras
        # tokenizer
        self.return_pt = paras.return_pt
        self.append_cls= paras.append_cls
        self.include_zero_gene= paras.include_zero_gene
        self.cls_token= paras.cls_token
        # pad
        self.my_vocab= vocab

        self.max_len= paras.max_len
        self.pad_token= paras.pad_token
        self.pad_value= paras.pad_value
        self.cls_appended= paras.cls_appended
        # mask
        self.mask_ratio= paras.mask_ratio
        self.mask_value= paras.mask_value
        self.pad_value= paras.pad_value

        self.init_special_tokens()

    def init_special_tokens(self):
        # special token
        self.special_tokens = [self.pad_token,self.cls_token,"<eoc>"]
        for s in self.special_tokens:
            if s not in self.my_vocab:
                self.my_vocab.append_token(s)

        self.cls_id = self.my_vocab.vocab[self.cls_token]

        
    def run(self,adata:AnnData,
                 ):
        input_layer_key = self.para.data_layer_name
        test_size = self.para.test_size
        shuffle = self.para.shuffle
        sort_seq_batch = self.para.sort_seq_batch
        # extend to matrixs
        matrixs = self.extend_to_matrixs(adata,input_layer_key)
        exist_genes = adata.var_names.tolist()
        # split train valid
        D_train,D_val = self.split_train_test(matrixs, test_size=test_size, shuffle=shuffle)

        # prepare data
        train_data_pt = self.prepare_data(D_train,gene_list=exist_genes,sort_seq_batch=sort_seq_batch)
        valid_data_pt = self.prepare_data(D_val,gene_list=exist_genes,sort_seq_batch=sort_seq_batch)

        # create dataset
        train_dataset = self.create_dataset(train_data_pt)
        valid_dataset = self.create_dataset(valid_data_pt)
        return train_dataset,valid_dataset,None
    ##############################################################################################################
    #  tokenize steps for scGPT
    #############################################################################################################
    def extend_to_matrixs(self,adata,input_layer_key:str = "X_binned"):
        
        all_counts = (
            adata.layers[input_layer_key].A
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )
        #genes = adata.var["gene_name"].tolist()
        cell_type_index = self.para.label_key
        celltypes_labels = adata.obs[cell_type_index].tolist()  # make sure count from 0
        num_types = len(set(celltypes_labels))
        celltypes_labels = np.array(celltypes_labels)

        batch_ids = adata.obs["batch_id"].tolist()
        num_batch_types = len(set(batch_ids))
        batch_ids = np.array(batch_ids)

        return [all_counts, celltypes_labels, num_types, batch_ids, num_batch_types]


    def split_train_test(self,matrixs, test_size=0.1, shuffle=True):
        [all_counts, celltypes_labels, num_types, batch_ids, num_batch_types] = matrixs
        from sklearn.model_selection import train_test_split
        (
            train_data,
            valid_data,
            train_celltype_labels,
            valid_celltype_labels,
            train_batch_labels,
            valid_batch_labels,
        ) = train_test_split(
            all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True)
        return [train_data,train_celltype_labels,train_batch_labels],[valid_data,valid_celltype_labels,valid_batch_labels]

    def prepare_data(self,
                     D,
                     gene_list,
                     sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
        [data, celltype_labels, batch_labels] = D
        # from vocab to gene_id numpy array
        self.my_vocab.set_default_index(self.my_vocab.vocab[self.pad_token])
        #gene_list= self.my_vocab.get_itos()
        #follows https://stackoverflow.com/questions/69015430/typeerror-vocab-object-is-not-callable to change
        #gene_ids = np.array(torch_vocab_item(gene_list), dtype=int)
        gene_ids = np.array(self.my_vocab.vocab.lookup_indices(gene_list), dtype=int)
        

        # 
        tokenized_data = self.tokenize_and_pad_batch(data,gene_ids)

        masked_values_train = self.random_mask_value(
            tokenized_data["values"],

        )
        show_txt = (masked_values_train == self.mask_value).sum() / (masked_values_train - self.pad_value).count_nonzero()
        logger.info(
            f"random masking at current epoch, ratio of masked values in train: {show_txt}"
        )
        input_gene_ids_train = tokenized_data["genes"]
        input_values_train = masked_values_train
        target_values_train = tokenized_data["values"]
        tensor_batch_labels = torch.from_numpy(batch_labels).long()
        

        if sort_seq_batch:
            train_sort_ids = np.argsort(batch_labels)
            input_gene_ids_train = input_gene_ids_train[train_sort_ids]
            input_values_train = input_values_train[train_sort_ids]
            target_values_train = target_values_train[train_sort_ids]
            tensor_batch_labels = tensor_batch_labels[train_sort_ids]


        data_pt = {
            "gene_ids": input_gene_ids_train,
            "values": input_values_train,
            "target_values": target_values_train,
            "batch_labels": tensor_batch_labels,
        }

        return data_pt

    def create_dataset(self,data_pt:Dict[str, torch.Tensor],)->Dataset:
        from scLLM.Dataset.dataset import SeqDataset
        dataset = SeqDataset(data_pt)
        return dataset



    def _tokenize_batch(
        self,
        data: np.ndarray,
        gene_ids: np.ndarray,
    ) -> List[Tuple[Union[torch.Tensor, np.ndarray]]]:
        """
        Tokenize a batch of data. Returns a list of tuple (gene_id, count).

        Args:
            data (array-like): A batch of data, with shape (batch_size, n_features).
                n_features equals the number of all genes.
            gene_ids (array-like): A batch of gene ids, with shape (n_features,).
            return_pt (bool): Whether to return torch tensors of gene_ids and counts,
                default to True.

        Returns:
            list: A list of tuple (gene_id, count) of non zero gene expressions.
        """
        if data.shape[1] != len(gene_ids):
            raise ValueError(
                f"Number of features in data ({data.shape[1]}) does not match "
                f"number of gene_ids ({len(gene_ids)})."
            )
        tokenized_data = []
        for i in range(len(data)):
            row = data[i]
            if self.include_zero_gene:
                values = row
                genes = gene_ids
            else:
                idx = np.nonzero(row)[0]
                values = row[idx]
                genes = gene_ids[idx]
            if self.append_cls:
                genes = np.insert(genes, 0, self.cls_id)
                values = np.insert(values, 0, 0)
            if self.return_pt:
                genes = torch.from_numpy(genes).long()
                values = torch.from_numpy(values)
            tokenized_data.append((genes, values))
        return tokenized_data


    def _pad_batch(
        self,
        batch: List[Tuple],

    ) -> Dict[str, torch.Tensor]:
        """
        Pad a batch of data. Returns a list of Dict[gene_id, count].

        Args:
            batch (list): A list of tuple (gene_id, count).
            max_len (int): The maximum length of the batch.
            vocab (Vocab): The vocabulary containing the pad token.
            pad_token (str): The token to pad with.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of gene_id and count.
        """
        vocab = self.my_vocab.to_torchtext_vocab()
        pad_id = vocab[self.pad_token]
        gene_ids_list = []
        values_list = []
        for i in range(len(batch)):
            gene_ids, values = batch[i]
            if len(gene_ids) > self.max_len:
                # sample max_len genes
                if not self.cls_appended:
                    idx = np.random.choice(len(gene_ids), self.max_len, replace=False)
                else:
                    idx = np.random.choice(len(gene_ids) - 1, self.max_len - 1, replace=False)
                    idx = idx + 1
                    idx = np.insert(idx, 0, 0)
                gene_ids = gene_ids[idx]
                values = values[idx]
            if len(gene_ids) < self.max_len:
                gene_ids = torch.cat(
                    [
                        gene_ids,
                        torch.full(
                            (self.max_len - len(gene_ids),), pad_id, dtype=gene_ids.dtype
                        ),
                    ]
                )
                values = torch.cat(
                    [
                        values,
                        torch.full((self.max_len - len(values),), self.pad_value, dtype=values.dtype),
                    ]
                )
            gene_ids_list.append(gene_ids)
            values_list.append(values)
        batch_padded = {
            "genes": torch.stack(gene_ids_list, dim=0),
            "values": torch.stack(values_list, dim=0),
        }
        return batch_padded


    def tokenize_and_pad_batch(
        self,
        data: np.ndarray,
        gene_ids: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize and pad a batch of data. Returns a list of tuple (gene_id, count).
        """
        tokenized_data = self._tokenize_batch(data,
                                            gene_ids,)
        batch_padded = self._pad_batch(tokenized_data)
        return batch_padded


    def random_mask_value(self,
        values: Union[torch.Tensor, np.ndarray],
        ) -> torch.Tensor:
        """
        Randomly mask a batch of data.

        Args:
            values (array-like):
                A batch of tokenized data, with shape (batch_size, n_features).
            mask_ratio (float): The ratio of genes to mask, default to 0.15.
            mask_value (int): The value to mask with, default to -1.
            pad_value (int): The value of padding in the values, will be kept unchanged.

        Returns:
            torch.Tensor: A tensor of masked data.
        """
        if isinstance(values, torch.Tensor):
            # it is crutial to clone the tensor, otherwise it changes the original tensor
            values = values.clone().detach().numpy()
        else:
            values = values.copy()

        for i in range(len(values)):
            row = values[i]
            non_padding_idx = np.nonzero(row - self.pad_value)[0]
            n_mask = int(len(non_padding_idx) * self.mask_ratio)
            mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
            row[mask_idx] = self.mask_value
        return torch.from_numpy(values).float()
    
    