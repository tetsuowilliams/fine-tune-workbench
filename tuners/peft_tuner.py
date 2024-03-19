import copy
from peft import LoraConfig, get_peft_model
from trainer import Trainer


class PEFTTuner:
    def __init__(self, device: str) -> None:
        self.device = device

    def train_model(self, trainer: Trainer):
        model_copy = copy.deepcopy(trainer.model)  
        
        config = LoraConfig(
            r=8,
            target_modules=["seq.0", "seq.2"],
            modules_to_save=["seq.4"],
        )

        peft_model = get_peft_model(model_copy, config)
        print(peft_model.print_trainable_parameters())

        trainer.train_and_test()
        