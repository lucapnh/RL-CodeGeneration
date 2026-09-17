import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer, get_peft_config


class SFTWrapper(SFTTrainer):
    """
    Thin subclass of TRL's SFTTrainer. The only reason this class exists is to make
    save_model() write adapters into a <output_dir>/policy/ subfolder, matching the
    checkpoint layout this repo's existing (unmodified) main.py / wrapper_utils.load_model
    eval path expects (savings/<model>/<dataset>/<checkpoint>/policy/). No training-loop
    logic is reimplemented here - SFTTrainer.train() runs unmodified.
    """

    def __init__(self, model_args, sft_args: SFTConfig, train_dataset, eval_dataset,
                 tokenizer, cache_dir: str, adapter_path: str = None):
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        )

        if adapter_path is not None:
            peft_config = None
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        else:
            peft_config = get_peft_config(model_args)

        super().__init__(
            model=model,
            args=sft_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir
        output_dir_policy = os.path.join(output_dir, "policy")
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir_policy)
        self.processing_class.save_pretrained(output_dir_policy)
