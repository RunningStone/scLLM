import attr 
from typing import Dict, Mapping, Optional, Tuple, Any, Union
from scLLM.Models.init import model_para_base

@attr.s(auto_attribs=True)
class scGPT_para(model_para_base):
    # transformer parameters
    #---> mode necessary
    ntoken: int = None
    d_model: int= None
    nhead: int= None
    d_hid: int= None
    nlayers: int= None
    #---> optional
    # how to embed raw data
    input_emb_style: str = "continuous" # "category" or "continuous" or "scaling"
    n_input_bins: Optional[int] = None # number of bins for input embedding if input_emb_style == "category": n_input_bins = n_bins + 2 else:n_input_bins = n_bins

    # which normalisation DSBN or BN
    domain_spec_batchnorm: Union[bool, str] = False #config.DSBN

    # which accelerated backend to use
    amp_flag: bool = False #config.amp_flag
    use_fast_transformer: bool = False
    fast_transformer_backend: str = "flash"

    # how to organise the normalisation step (before or after the attention)
    pre_norm: bool = False

    #--->data preprocess related
    vocab: Any = None
    dropout: float = 0.5
    mask_value = -1
    pad_token: str = "<pad>"
    pad_value: int = 0
    #--->train related
    use_batch_labels: bool = False
    num_batch_labels: Optional[int] = None
    #--->decoder general
    explicit_zero_prob: bool = False # whether explicit bernoulli for zeros

    #--> specify forward steps and params
    #masked language modeling
    MLM = False  # whether to use masked language modeling, currently it is always on.

    #Domain adaptation by reverse backpropagation: the objective weight for batch correction (DAR)
    DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
    do_dab: bool = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
    dab_weight: float = 1.0

    # classification 
    CLS: bool = False # celltype classification objective
    nlayers_cls: int = 3
    n_cls: int = 1 #num_types if CLS else 1, num_types = len(np.unique(celltype_id_labels))
    cell_emb_style: str = "cls"

    # Contrastive cell embedding objective
    CCE: bool = False

    # Masked value prediction for cell embedding # Gene expression modelling for cell objective
    do_mvc: bool = False #config.GEPC, init MVC sub-net 
    MVC: bool = False    # Masked value prediction for cell embedding
    mvc_decoder_style: str = "inner product"

    # Elastic cell similarity objective
    ECS: bool = False
    ecs_threshold: float = 0.3 #config.ecs_thres,

    do_sample: bool = False


    #--> specified ops
    ops_class_name:list = ["custom_norm","flash_attention"]