import torch
import torch.nn.functional as F
from typing import List
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

def tokenize_prompt_and_output(prompt_strs: List[str], output_strs: List[str], tokenizer: PreTrainedTokenizerBase):
    batch_tokens = []
    batch_mask = []
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_stripped = prompt.strip()
        output_stripped = output.strip()

        clean_output = output_stripped
        if prompt_stripped.endswith('<think>') and output_stripped.startswith('<think>'):
            clean_output = output_stripped[7:].lstrip()

        prompt_ids = tokenizer.encode(prompt_stripped, add_special_tokens = False)
        output_ids = tokenizer.encode(clean_output, add_special_tokens = False)
        eos_id = tokenizer.eos_token_id
        assert isinstance(eos_id, int)
        if tokenizer.eos_token_id is not None:
            output_ids.append(eos_id)
        else:
            pass
        token_ids = prompt_ids + output_ids
        output_mask = [0] * len(prompt_ids) + [1] * len(output_ids)
        batch_tokens.append(torch.tensor(token_ids, dtype=torch.long))
        batch_mask.append(torch.tensor(output_mask, dtype=torch.long))

    max_len = max(len(ids) for ids in batch_tokens)
    padded_input_ids = []
    padded_masks = []
    padded_labels = []
    for input_ids, masks in zip(batch_tokens, batch_mask):
        pad_len = max_len - len(input_ids)
        padded_input = F.pad(input=input_ids, pad=(0, pad_len), value=tokenizer.pad_token_id) # 对最后一个dim,左填充0个，右填充pad_len个
        padded_mask = F.pad(input=masks, pad=(0, pad_len), value=0)

        padded_input_ids.append(padded_input)
        padded_masks.append(padded_mask)

    batch_input_ids = torch.stack(padded_input_ids)
    batch_masks = torch.stack(padded_masks)

    shifted_inputs = batch_input_ids[:, :-1]
    shifted_masks = batch_masks[:, 1:]
    labels = batch_input_ids[:, 1:]

    return {
        "input_ids": shifted_inputs,
        "labels": labels,
        "response_masks": shifted_masks
    }

