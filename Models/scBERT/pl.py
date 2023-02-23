

from scLLM.Trainer.base import pl_basic

class pl_scBERT(pl_basic):
    def __init__(self,
                    trainer_paras, # trainer para
                    model_paras, # model para
                    ):
            super(pl_scBERT,self).__init__(trainer_paras,model_paras)
            print("init scBERT part...")
            self.trainer_paras.init_optimizers()

    def create_model(self):
        """
        create model instance
        """
        from scLLM.Models.scBERT.model import PerformerLM
        self.model = PerformerLM(self.model_paras)


        