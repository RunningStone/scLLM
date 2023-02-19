from typing import Any, Optional, Union

class scFormer_para:
    # need to set when init
    ntoken: int = None,
    dim: int = None,
    heads: int = None,
    dim_head: int = None,
    depth: int = None,
    nlayers_cls: int= None,
    n_cls: int= None,
    vocab: Any= None,
    # default
    dropout: float = 0.5,
    pad_token: str = "<pad>",
    pad_value: int = 0,
    do_mvc: bool = False,
    do_dab: bool = False,
    use_batch_labels: bool = False,
    num_batch_labels: Optional[int] = None,
    domain_spec_batchnorm: Union[bool, str] = False,
    input_emb_style: str = "continuous",
    n_input_bins: Optional[int] = None,
    cell_emb_style: str = "cls",
    mvc_decoder_style: str = "inner product",
    ecs_threshold: float = 0.3,
    explicit_zero_prob: bool = False,