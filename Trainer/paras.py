from typing import Any, Optional, Union

class Trainer_para:
    # need to set when init
    #-----> dataset

    #-----> model
    ckpt_loc: str = None

    #-----> training
    max_epochs: int = 100