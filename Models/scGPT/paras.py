import attr 
from typing import Dict, Mapping, Optional, Tuple, Any, Union

@attr.s(auto_attribs=True)
class scGPT_para:
    # transformer parameters
    #---> necessary
    ntoken: int = None
    d_model: int= None
    nhead: int= None
    d_hid: int= None
    nlayers: int= None
    #---> optional
    nlayers_cls: int = 3
    n_cls: int = 1
    vocab: Any = None
    dropout: float = 0.5
    pad_token: str = "<pad>"
    pad_value: int = 0
    do_mvc: bool = False #config.GEPC,
    do_dab: bool = False
    use_batch_labels: bool = False
    num_batch_labels: Optional[int] = None
    domain_spec_batchnorm: Union[bool, str] = False #config.DSBN
    input_emb_style: str = "continuous"

    n_input_bins: Optional[int] = None
    cell_emb_style: str = "cls"
    mvc_decoder_style: str = "inner product"
    ecs_threshold: float = 0.3 #config.ecs_thres,
    explicit_zero_prob: bool = False
    use_fast_transformer: bool = False
    fast_transformer_backend: str = "flash"
    pre_norm: bool = False