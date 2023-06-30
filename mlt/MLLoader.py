from pathlib import Path

from peft import PeftModel

from mlt.MLTrainer import set_additional_trainable_modules


def loadMLModel(model, loras: list, save_dir: str,):
    """
    Load a model from a pretrained model and a list of loras.

    :param model: A pretrained model.
    :param loras: A list of loras.
    :param save_dir: Directory where the loras are saved.

    :return: A model with the loras loaded.
    """
    if loras is None or len(loras) == 0:
        print("No loras to load.")
        return model
    if not isinstance(model, PeftModel):
        model: PeftModel = PeftModel.from_pretrained(model, Path(f"{save_dir}/{loras[0]}"),
                                                     adapter_name=loras[0])
    else:
        model.load_adapter(Path(f"{save_dir}/{loras[0]}"), loras[0])
    set_additional_trainable_modules(model, loras[0])   # set additional trainable modules to avoid errors during training
    for lora in loras[1:]:
        model.load_adapter(Path(f"{save_dir}/{lora}"), lora)
        set_additional_trainable_modules(model, lora)    # set additional trainable modules to avoid errors during training
    return model
