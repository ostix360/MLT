from pathlib import Path

from peft import PeftModel




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


def set_additional_trainable_modules(model, lora_name):
    """
    Sets additional trainable modules for a given adapter.

    Useful for classification models
        classifier layer is added as trainable module for the adapter
        Necessary for training to avoid a torch error
    :param model: Model
    :param str lora_name: the lora adapter name to add
    :return: None
    """
    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        target_module_found = any(key.endswith(target_key) for target_key in model.modules_to_save)
        if target_module_found:
            parent, target, target_name = _get_submodules(model, key)
            if isinstance(target, ModulesToSaveWrapper):
                target.update(lora_name)
            else:
                setattr(parent, target_name, ModulesToSaveWrapper(target, lora_name))
            for param in target.parameters():
                param.requires_grad = True
