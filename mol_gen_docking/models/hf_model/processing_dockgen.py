"""To minimize modifications to pre-existing code,
we store all multimodal informations as image embeddings.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, cast

import torch
from transformers import PreTrainedTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import (
    AllKwargsForChatTemplate,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
    render_jinja_template,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

JINJA_TEMPLATE = """
{%- set image_count = namespace(value=0) %}\
{%- set content_ns_sys = namespace(content='') %}
{%- if messages[0].role == \'system\' %}\
    {%- if messages[0].content is string %}
        {%- set content_ns_sys.content = messages[0].content %}
    {%- elif 'type' in messages[0].content[0] %}
        {%- set content_ns_sys.content = messages[0].content[0].text %}
        {%- endif %}
    {%- endif %}
{%- if tools %}\
    {{- \'<|im_start|>system\\n\' }}\
    {%- if messages[0].role == \'system\' %}\
        {{- content_ns_sys.content ~ \'\\n\\n\' }}\
        {%- endif %}\
    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}\
    {%- for tool in tools %}\
        {{- "\\n" }}\
        {{- tool | tojson }}\
        {%- endfor %}\
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}\
{%- else %}\
    {%- if messages[0].role == \'system\' %}\
        {{- \'<|im_start|>system\\n\' ~ content_ns_sys.content ~ \'<|im_end|>\\n\' }}\
        {%- endif %}\
    {%- endif %}\
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\
{%- for message in messages[::-1] %}\
    {%- set index = (messages|length - 1) - loop.index0 %}\
    {%- if ns.multi_step_tool and message.role == "user" and (message.content is string or 'text' in message.content[0]) %}\
        {%- set text_content = message.content if message.content is string else message.content[0].text %}\
        {%- if not text_content.startswith('<tool_response>') and not text_content.endswith('</tool_response>') %}\
            {%- set ns.multi_step_tool = false %}\
            {%- set ns.last_query_index = index %}\
            {%- endif %}\
        {%- endif %}\
    {%- endfor %}\
{%- for message in messages %}\
    {%- set content_ns = namespace(content='') %}
    {%- if message.content is string %}\
        {%- set content_ns.content = message.content %}\
    {%- elif 'type' in message.content[0] %}\
        {%- for m in message.content %}\
            {%- if 'text' in m %}\
                {%- set content_ns.content = content_ns.content ~ m.text %}\
                {%- endif %}\
            {%- endfor %}\
        {%- endif %}\
    {%- set content = content_ns.content %}\
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}\
        {{- \'<|im_start|>\' ~ message.role ~ \'\\n\' ~ content ~ \'<|im_end|>\' ~ \'\\n\' }}\
    {%- elif message.role == "assistant" %}\
        {%- set reasoning_content = \'\' %}\
        {%- if message.reasoning_content is string %}\
            {%- set reasoning_content = message.reasoning_content %}\
        {%- else %}\
            {%- if \'</think>\' in content %}\
                {%- set reasoning_content = content.split(\'</think>\')[0].rstrip(\'\\n\').split(\'<think>\')[-1].lstrip(\'\\n\') %}\
                {%- set content = content.split(\'</think>\')[-1].lstrip(\'\\n\') %}\
                {%- endif %}\
            {%- endif %}\
        {%- if loop.index0 > ns.last_query_index %}\
            {%- if loop.last or (not loop.last and reasoning_content) %}\
                {{- \'<|im_start|>\' ~ message.role ~ \'\\n<think>\\n\' ~ reasoning_content.strip(\'\\n\') ~ \'\\n</think>\\n\\n\' ~ content.lstrip(\'\\n\') }}\
            {%- else %}\
                {{- \'<|im_start|>\' ~ message.role ~ \'\\n\' ~ content }}\
                {%- endif %}\
        {%- else %}\
            {{- \'<|im_start|>\' ~ message.role ~ \'\\n\' ~ content }}\
            {%- endif %}\
        {%- if message.tool_calls %}\
            {%- for tool_call in message.tool_calls %}\
                {%- if (loop.first and content) or (not loop.first) %}\
                    {{- \'\\n\' }}\
                    {%- endif %}\
                {%- if tool_call.function %}\
                    {%- set tool_call = tool_call.function %}\
                    {%- endif %}\
                {{- \'<tool_call>\\n{"name": "\' }}\
                {{- tool_call.name }}\
                {{- \'", "arguments": \' }}\
                {%- if tool_call.arguments is string %}\
                    {{- tool_call.arguments }}\
                {%- else %}\
                    {{- tool_call.arguments | tojson }}\
                    {%- endif %}\
                {{- \'}\\n</tool_call>\' }}\
                {%- endfor %}\
            {%- endif %}\
        {{- \'<|im_end|>\\n\' }}\
    {%- elif message.role == "tool" %}\
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}\
            {{- \'<|im_start|>user\' }}\
            {%- endif %}\
        {{- \'\\n<tool_response>\\n\' }}\
        {{- content }}\
        {{- \'\\n</tool_response>\' }}\
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}\
            {{- \'<|im_end|>\\n\' }}\
            {%- endif %}\
        {%- endif %}\
    {%- endfor %}\
{%- if add_generation_prompt %}\
    {{- \'<|im_start|>assistant\\n\' }}\
    {%- if enable_thinking is defined and enable_thinking is false %}\
        {{- \'<think>\\n\\n</think>\\n\\n\' }}\
        {%- endif %}\
    {%- endif %}
"""


@dataclass
class ProtKwargs: ...


class DockGenProcessorKwargs(ProcessingKwargs):
    images_kwargs: ProtKwargs = ProtKwargs()
    _defaults = {
        "text_kwargs": {
            "padding": "longest",
            "return_tensors": "pt",
        },
        "images_kwargs": {},
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }


class DockGenProcessor(ProcessorMixin):
    """
    A tokenizer for the Qwen-3 model that handles multimodal inputs.

    The input of the tokenizer is expected to be a list of dictionaries,
    where each dictionaries containing a 'text' keya are separated by a special token: ""<|image_pad|>".
    """

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        image_token: str = "<|image_pad|>",
        **kwargs: Unpack[DockGenProcessorKwargs],
    ) -> None:
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        self.image_token = (
            tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        )
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )

        super().__init__(tokenizer)
        self.chat_template = (
            kwargs["chat_template"] if "chat_template" in kwargs else JINJA_TEMPLATE
        )

    def __call__(
        self,
        images: Optional[List[torch.Tensor]] = None,
        text: Union[
            TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]
        ] = None,
        **kwargs: Unpack[DockGenProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            DockGenProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        return_tensors = output_kwargs["common_kwargs"]["return_tensors"]

        if images is not None:
            if len(images) == 1:
                image_inputs = {
                    "pixel_values": images[0].unsqueeze(0)
                }  # SHAPE: [BATCH_SIZE, MAX_N_MM, EMB_SIZE]
            else:
                max_n_mm = max([val.shape[0] for val in images])
                image_inputs = {
                    "pixel_values": torch.stack(
                        [
                            torch.cat(
                                [
                                    val,
                                    torch.zeros(
                                        max_n_mm - val.shape[0], *val.shape[1:]
                                    ),
                                ]
                            )
                            for val in images
                        ],
                        dim=0,
                    )
                }
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        # try to expand inputs in processing if we have the necessary parts
        prompt_strings = text

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])

        self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])

        return BatchFeature(
            data={
                **text_inputs,
                **image_inputs,
            },
            tensor_type=return_tensors,
        )

    # Need to override apply_chat_template
    def apply_chat_template(
        self,
        conversation: List[List[Dict[str, Any]]] | List[Dict[str, Any]],
        chat_template: Optional[str] = None,
        **kwargs: Unpack[AllKwargsForChatTemplate],
    ) -> Any:
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
            if (
                isinstance(self.chat_template, dict)
                and chat_template in self.chat_template
            ):
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
                raise ValueError(
                    "continue_final_message is not compatible with return_assistant_tokens_mask."
                )

        # Fill sets of kwargs that should be used by different parts of template
        processed_kwargs: Dict[str, Dict[str, Any]] = {
            "mm_load_kwargs": {},
            "template_kwargs": {},
        }

        for kwarg_type in processed_kwargs:
            for key in AllKwargsForChatTemplate.__annotations__[
                kwarg_type
            ].__annotations__.keys():
                kwarg_type_defaults = AllKwargsForChatTemplate.__annotations__[
                    kwarg_type
                ]
                default_value = getattr(kwarg_type_defaults, key, None)
                value = kwargs.pop(key, default_value)
                if value is not None and not isinstance(value, dict):
                    processed_kwargs[kwarg_type][key] = value

        # Pass unprocessed custom kwargs
        processed_kwargs["template_kwargs"].update(kwargs)

        conversations: List[List[Dict[str, Any]]] = []
        if isinstance(conversation, list) and isinstance(conversation[0], list):
            conversations = cast(List[List[Dict[str, Any]]], conversation)
            is_batched = True
        elif isinstance(conversation, list) and isinstance(conversation[0], dict):
            conversations = [cast(List[Dict[str, Any]], conversation)]
            is_batched = False
        else:
            raise ValueError("Wrong conversation format.")

        tokenize = processed_kwargs["template_kwargs"].pop("tokenize", False)
        return_dict = processed_kwargs["template_kwargs"].pop("return_dict", False)
        # mm_load_kwargs = processed_kwargs["mm_load_kwargs"]

        if tokenize:
            batch_images: List[torch.Tensor] = []
            for conv in conversations:
                images: List[torch.Tensor] = []
                for message in conv:
                    visuals: List[Dict[str, Any]] = [
                        content
                        for content in message["content"]
                        if content["type"] in ["image"]
                    ]
                    image_fnames: List[str] = [
                        vision_info["path"]
                        for vision_info in visuals
                        if vision_info["type"] == "image"
                    ]

                    for fname in image_fnames:
                        # TODO: Get embeddings
                        data_path = os.environ.get("DATA_PATH", "")

                        images.append(torch.load(os.path.join(data_path, fname)))
                if images:
                    batch_images.append(torch.stack(images, dim=0))

        prompt, generation_indices = render_jinja_template(
            conversations=conversations,
            chat_template=chat_template,
            **processed_kwargs[
                "template_kwargs"
            ],  # different flags such as `return_assistant_mask`
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
            if self.tokenizer.bos_token is not None and single_prompt.startswith(
                self.tokenizer.bos_token
            ):
                kwargs["add_special_tokens"] = False

            out = self(
                text=prompt,
                images=batch_images if batch_images else None,
                **kwargs,
            )
            if return_dict:
                if processed_kwargs["template_kwargs"].get(
                    "return_assistant_tokens_mask", False
                ):
                    assistant_masks = []
                    input_ids = out["input_ids"]
                    for i in range(len(input_ids)):
                        current_mask = [0] * len(input_ids[i])
                        for (
                            assistant_start_char,
                            assistant_end_char,
                        ) in generation_indices[i]:
                            start_token = out.char_to_token(i, assistant_start_char)
                            end_token = out.char_to_token(i, assistant_end_char - 1)
                            if start_token is None:
                                # start_token is out of bounds maybe due to truncation.
                                break
                            for token_id in range(
                                start_token,
                                end_token + 1 if end_token else len(input_ids[i]),
                            ):
                                current_mask[token_id] = 1
                        assistant_masks.append(current_mask)
                    out["assistant_masks"] = assistant_masks
                    out.convert_to_tensors(
                        tensor_type=kwargs.get("return_tensors", None)
                    )
                return out
            else:
                return out["input_ids"]
        return prompt

    def batch_decode(self, *args: Any, **kwargs: Any) -> Any:
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any) -> Any:
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)
