import argparse
import json
import os

from transformers import AutoTokenizer
from trl import ModelConfig, ScriptArguments, SFTConfig

from sft.sft_dataset_utils import get_robo_instruct_sft_dataset
from sft.sft_wrapper import SFTWrapper

# Cache location for HuggingFace models/datasets, same convention as main.py.
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def parse_sft_hyperparams(path: str):
    params = json.load(open(path))
    script_args = ScriptArguments(**params["ScriptArguments"])
    model_args = ModelConfig(**params["ModelConfig"])
    sft_args = SFTConfig(
        # "sft" subdirectory keeps this fully separate from PPO's checkpoints, which
        # live directly under savings/<model>/<dataset>/ - while still resolving
        # correctly when the original main.py is given `--checkpoint sft/checkpoint-N`.
        output_dir=os.path.join("savings", model_args.model_name_or_path, script_args.dataset_name, "sft"),
        logging_dir=os.path.join("savings", "logs", model_args.model_name_or_path, script_args.dataset_name, "sft"),
        **params["SFTConfig"],
    )
    return script_args, model_args, sft_args


if __name__ == "__main__":
    os.environ["TRANSFORMERS_CACHE"] = HF_HOME
    os.environ["HF_HOME"] = HF_HOME
    os.environ["HF_HUB_CACHE"] = HF_HOME

    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperparams", type=str, default=os.path.join("sft", "hyperparams.json"))
    parser.add_argument("--max_steps", type=int, default=None,
                         help="Override total training steps, for smoke-testing (e.g. --max_steps 1). "
                              "Leave unset for a real run driven by num_train_epochs.")
    parser.add_argument("--adapter_path", type=str, default=None,
                         help="Resume training from an existing adapter checkpoint's policy/ directory.")
    args = parser.parse_args()

    script_args, model_args, sft_args = parse_sft_hyperparams(args.hyperparams)
    if args.max_steps is not None:
        sft_args.max_steps = args.max_steps

    dataset = get_robo_instruct_sft_dataset(model_args.model_name_or_path, cache_dir=HF_HOME)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=HF_HOME)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    trainer = SFTWrapper(
        model_args=model_args,
        sft_args=sft_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        cache_dir=HF_HOME,
        adapter_path=args.adapter_path,
    )
    trainer.train()
    # Save under a checkpoint-N subfolder (matching Trainer's own mid-training
    # checkpoint layout) so the final adapter is reachable via the same
    # `--checkpoint sft/checkpoint-N` convention as any intermediate checkpoint.
    final_checkpoint_dir = os.path.join(sft_args.output_dir, f"checkpoint-{trainer.state.global_step}")
    trainer.save_model(final_checkpoint_dir)
