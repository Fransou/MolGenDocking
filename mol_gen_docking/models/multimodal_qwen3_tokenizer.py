"""To minimize modifications to pre-existing code,
we store all multimodal informations as image embeddings.
"""
from dataclasses import dataclass
import numpy as np
import torch
from typing import List, Optional, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, _validate_images_text_input_order, AllKwargsForChatTemplate, render_jinja_template
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

@dataclass
class PdbProcessorKwargs:
    ...

class TextPdbProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: PdbProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "padding": "max_length",
        },
        "images_kwargs": {},
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }

class TextPdbProcessor(ProcessorMixin):
    """
    A tokenizer for the Qwen-3 model that handles multimodal inputs.

    The input of the tokenizer is expected to be a list of dictionaries,
    where each dictionaries containing a 'text' keya are separated by a special token: "<|prot|>".
    """

    attributes = ["image_processor", "tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None,image_token="<|PROT|>", **kwargs):

        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        super().__init__(image_processor, tokenizer)

    def __call__(
            self,
            images: List[str], # PDB Path
            text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
            audio=None,
            videos=None,
            **kwargs: Unpack[TextPdbProcessorKwargs],
    ) -> BatchFeature:

        output_kwargs = self._merge_kwargs(
            TextPdbProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        return_tensors = output_kwargs["common_kwargs"]["return_tensors"]

        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        # try to expand inputs in processing if we have the necessary parts
        prompt_strings = text

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])

        input_ids = text_inputs["input_ids"]
        special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)

        self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])

        return BatchFeature(data={"special_image_mask":special_image_mask, **text_inputs, **image_inputs}, tensor_type=return_tensors)


# Need to override apply_chat_template
    def apply_chat_template(
            self,
            conversation: Union[list[dict[str, str]], list[list[dict[str, str]]]],
            chat_template: Optional[str] = None,
            **kwargs: Unpack[AllKwargsForChatTemplate],
    ) -> str:
        if chat_template is None:
            if isinstance(self.chat_template, dict) and "default" in self.chat_template:
                chat_template = self.chat_template["default"]
            elif isinstance(self.chat_template, dict):
                raise ValueError(
                    'The processor has multiple chat templates but none of them are named "default". You need to specify'
                    " which one to use by passing the `chat_template` argument. Available templates are: "
                    f"{', '.join(self.chat_template.keys())}"
                )
            elif self.chat_template is not None:
                chat_template = self.chat_template
            else:
                raise ValueError(
                    "Cannot use apply_chat_template because this processor does not have a chat template."
                )
        else:
            if isinstance(self.chat_template, dict) and chat_template in self.chat_template:
                # It's the name of a template, not a full template string
                chat_template = self.chat_template[chat_template]
            else:
                # It's a template string, render it directly
                chat_template = chat_template

        if kwargs.get("continue_final_message", False):
            if kwargs.get("add_generation_prompt", False):
                raise ValueError(
                    "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead."
                )
            if kwargs.get("return_assistant_tokens_mask", False):
                raise ValueError("continue_final_message is not compatible with return_assistant_tokens_mask.")

        # Fill sets of kwargs that should be used by different parts of template
        processed_kwargs = {
            "mm_load_kwargs": {},
            "template_kwargs": {},
        }

        for kwarg_type in processed_kwargs:
            for key in AllKwargsForChatTemplate.__annotations__[kwarg_type].__annotations__.keys():
                kwarg_type_defaults = AllKwargsForChatTemplate.__annotations__[kwarg_type]
                default_value = getattr(kwarg_type_defaults, key, None)
                value = kwargs.pop(key, default_value)
                if value is not None and not isinstance(value, dict):
                    processed_kwargs[kwarg_type][key] = value

        # Pass unprocessed custom kwargs
        processed_kwargs["template_kwargs"].update(kwargs)

        if isinstance(conversation, (list, tuple)) and (
                isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "content")
        ):
            is_batched = True
            conversations = conversation
        else:
            is_batched = False
            conversations = [conversation]

        tokenize = processed_kwargs["template_kwargs"].pop("tokenize", False)
        return_dict = processed_kwargs["template_kwargs"].pop("return_dict", False)
        mm_load_kwargs = processed_kwargs["mm_load_kwargs"]

        if tokenize:
            batch_images = []
            for conversation in conversations:
                images = []
                for message in conversation:
                    visuals = [content for content in message["content"] if content["type"] in ["image"]]
                    image_fnames = [
                        vision_info["path"]
                        for vision_info in visuals
                        if vision_info["type"] == "image"
                    ]

                    for fname in image_fnames:
                        # TODO: Get embeddings
                        images.append(torch.load(fname))
                if images:
                    batch_images.append(images)

        prompt, generation_indices = render_jinja_template(
            conversations=conversations,
            chat_template=chat_template,
            **processed_kwargs["template_kwargs"],  # different flags such as `return_assistant_mask`
            **self.tokenizer.special_tokens_map,  # tokenizer special tokens are used by some templates
        )

        if not is_batched:
            prompt = prompt[0]

        if tokenize:
            # Tokenizer's `apply_chat_template` never adds special tokens when tokenizing
            # But processor's `apply_chat_template` didn't have an option to tokenize, so users had to format the prompt
            # and pass it to the processor. Users thus never worried about special tokens relying on processor handling
            # everything internally. The below line is to keep BC for that and be able to work with model that have
            # special tokens in the template (consistent with tokenizers). We dont want to raise warning, it will flood command line
            # without actionable solution for users
            single_prompt = prompt[0] if is_batched else prompt
            if self.tokenizer.bos_token is not None and single_prompt.startswith(self.tokenizer.bos_token):
                kwargs["add_special_tokens"] = False

            out = self(
                text=prompt,
                images=batch_images if batch_images else None,
                **kwargs,
            )
            if return_dict:
                if processed_kwargs["template_kwargs"].get("return_assistant_tokens_mask", False):
                    assistant_masks = []
                    input_ids = out["input_ids"]
                    for i in range(len(input_ids)):
                        current_mask = [0] * len(input_ids[i])
                        for assistant_start_char, assistant_end_char in generation_indices[i]:
                            start_token = out.char_to_token(i, assistant_start_char)
                            end_token = out.char_to_token(i, assistant_end_char - 1)
                            if start_token is None:
                                # start_token is out of bounds maybe due to truncation.
                                break
                            for token_id in range(start_token, end_token + 1 if end_token else len(input_ids[i])):
                                current_mask[token_id] = 1
                        assistant_masks.append(current_mask)
                    out["assistant_masks"] = assistant_masks
                    out.convert_to_tensors(tensor_type=kwargs.get("return_tensors", None))
                return out
            else:
                return out["input_ids"]
        return prompt

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

