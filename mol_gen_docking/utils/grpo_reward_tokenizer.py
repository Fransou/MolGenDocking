"""Wrapps a normal tokenizer to make sure its batch decode does not skip special tokens"""

from tokenizers import Tokenizer
from copy import deepcopy


def wrap_tokenizer(tokenizer: Tokenizer) -> Tokenizer:
    """
    Wraps a normal tokenizer to make sure its batch_decode does not skip special tokens
    """
    new_tokenizer = deepcopy(tokenizer)

    old_batch_decode = new_tokenizer.batch_decode

    def batch_decode(*args, **kwargs):
        """
        Batch decode sequences
        """
        kwargs["skip_special_tokens"] = False
        out = old_batch_decode(*args, **kwargs)
        out = [x.replace(new_tokenizer.special_token_map["pad_token"], "") for x in out]
        return out

    new_tokenizer.batch_decode = batch_decode
    return new_tokenizer
