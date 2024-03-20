import copy
from peft import LoraConfig, get_peft_model
from peft.peft_model import PeftModel


class PeftBuilder:
    def __init__(self, device: str) -> None:
        self.device = device
        self.model_copy = None

    def get_model(self, model) -> PeftModel:
        self.model_copy = copy.deepcopy(model)  
        
        config = LoraConfig(
            r=8,
            target_modules=["seq.0", "seq.2"],
            modules_to_save=["seq.4"],
        )

        peft_model = get_peft_model(self.model_copy, config)
        print(peft_model.print_trainable_parameters())
        return peft_model
        