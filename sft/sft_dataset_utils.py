from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from rl_datasets.dataset_special_functions import robo_instruct_tokenization_function


def get_robo_instruct_sft_dataset(model_or_tokenizer_name: str, cache_dir: str, seed: int = 42) -> DatasetDict:
    """
    Builds a single-column ("text") causal-LM SFT dataset from zichao22/robo-instruct.

    Reuses robo_instruct_tokenization_function (rl_datasets/dataset_special_functions.py)
    as-is to render the few-shot chat-template prompt and clean the target code - the same
    function the PPO training path uses, so this matches PPO's training-time prompt format.

    """
    raw_dataset = load_dataset("zichao22/robo-instruct", cache_dir=cache_dir)["train"]
    split = raw_dataset.train_test_split(test_size=0.2, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(model_or_tokenizer_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    def format_split(dataset):
        input_sequences = list(dataset["text"])
        target_sequences = list(dataset["program"])

        # Mutates input_sequences/target_sequences in place: renders the few-shot
        # chat-template prompt and strips NEW_LINE/INDENT/DEDENT from the target code.
        robo_instruct_tokenization_function(input_sequences, target_sequences, tokenizer)

        # Newline after ```python matches the few-shot demonstration's fencing
        # style (its opening fence has one baked into the source example text);
        # no newline before the closing ``` matches that same precedent too.
        texts = [
            prompt + "```python\n" + code + "```" + tokenizer.eos_token
            for prompt, code in zip(input_sequences, target_sequences)
        ]
        return Dataset.from_dict({"text": texts})

    return DatasetDict({
        "train": format_split(split["train"]),
        "test": format_split(split["test"]),
    })
