from typing import Any, Optional, Union
import attr

@attr.s(auto_attribs=True)
class Dataset_para:
    """
    Dataset parameters
    """
    #--------> data loading steps
    # for gene2vec encoding or other embedding methods the entire vocabulary is needed
    gene_vocab:list = None #['gene1','gene2',...] 
    #--------> preprocessing steps
    #(:class:`str`, optional) The key of :class:`~anndata.AnnData` to use for preprocessing.
    use_key: Optional[str] = "X" 
    
    #---> gene filter and cell filter
    #filter_gene_by_counts (:class:`int` or :class:`bool`, default: ``False``): Whther to filter genes by counts, if :class:`int`, filter genes with counts
    filter_gene_by_counts: Union[int, bool] = False #False
    #filter_cell_by_counts (:class:`int` or :class:`bool`, default: ``False``): Whther to filter cells by counts, if :class:`int`, filter cells with counts
    filter_cell_by_counts: Union[int, bool] = 200#False
    
    #----> normalization
    #(:class:`float` or :class:`bool`, default: ``1e4``): Whether to normalize the total counts of each cell to a specific value.
    normalize_total: Union[float, bool] = 1e4
    #(:class:`str`, default: ``"X_normed"``): The key of :class:`~anndata.AnnData` to store the normalized data. If :class:`None`, will use normed data to replce the :attr:`use_key`.
    result_normed_key: Optional[str] = "X_normed"
    #---> log1p transform
    #(:class:`bool`, default: ``True``): Whether to apply log1p transform to the normalized data.
    log1p: bool = True #False
    #(:class:`str`, default: ``"X_log1p"``): The key of :class:`~anndata.AnnData` to store the log1p transformed data.
    result_log1p_key: str = "X_log1p"
    #(:class:`float`, default: ``2.0``): The base para of log1p transform funciton.
    log1p_base: float = 2.0
    #--->hvg
    #(:class:`int` or :class:`bool`, default: ``False``): Whether to subset highly variable genes.
    subset_hvg: Union[int, bool] = False
    #(:class:`str`, optional): The key of :class:`~anndata.AnnData` to use for calculating highly variable genes. If :class:`None`, will use :attr:`adata.X`.
    hvg_use_key: Optional[str] = None
    #(:class:`str`, default: ``"seurat_v3"``): The flavor of highly variable genes selection. See :func:`scanpy.pp.highly_variable_genes` for more details.
    hvg_flavor: str = "seurat_v3"
    #---->bined data part
    #(:class:`int`, optional): Whether to bin the data into discrete values of number of bins provided.
    binning: Optional[int] = None
    #(:class:`str`, default: ``"X_binned"``): The key of :class:`~anndata.AnnData` to store the binned data.
    result_binned_key: str = "X_binned"

    #--------> data saving steps
    preprocessed_loc: str = None

    #--------> dataset steps

    #--------> dataloader steps
    batch_size: int = 32
    shuffle:bool =True
    num_workers:int =0
    additional_dataloader_para:dict = {}