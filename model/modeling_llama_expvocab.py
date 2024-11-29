from typing import Optional, Union, List

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    LlamaForCausalLM,
    LlamaModel,
    LlamaConfig,
    AutoTokenizer,
    Cache,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.modeling_outputs import CausalLMOutputWithPast


CACHE_DIR = "/workspace/.cache"
LLAMA_PAD_ID = 128009 # "content": "<|eot_id|>"
GPT2_PAD_ID = 55025   # `SEPARATOR` in anticipation
GPT2_VOCAB_SIZE = 55030

class LlamaWithExpandedVocabModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if hasattr(config, "expanded_vocab_size"):
            self.expanded_vocab_size = config.expanded_vocab_size
            self.expanded_embed_tokens = nn.Embedding(config.expanded_vocab_size, config.hidden_size, GPT2_PAD_ID)
            self.expanded_embed_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    # TODO: wire the expanded vocab into the forward pass
    # def forward(...)


class LlamaWithExpandedVocabForCausalLM(LlamaForCausalLM):
    def __init__(self, config, expanded_vocab_size=None):
        super().__init__(config)

        if expanded_vocab_size is not None or hasattr(config, "expanded_vocab_size"):
            self.expanded_vocab_size = expanded_vocab_size or config.expanded_vocab_size

            if not hasattr(config, "expanded_vocab_size"):
                config.expanded_vocab_size = self.expanded_vocab_size

        self.model = LlamaWithExpandedVocabModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if self.expanded_vocab_size is not None:
            self.expanded_lm_head = nn.Linear(config.hidden_size, self.expanded_vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        assert input_ids is not None

        # create a (0, 1) mask to indicate which tokens are in the original vocabulary
        orig_vocab_positions = torch.lt(input_ids, self.vocab_size).to(
            device=input_ids.device
        )
        orig_vocab_positions.requires_grad_(False)
        # left shift by 1 to account for next-token prediction
        orig_vocab_positions = torch.roll(orig_vocab_positions, shifts=-1, dims=-1)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

            # mask out logits for tokens in the expanded vocabulary with -inf
            logits = logits.masked_fill(
                ~orig_vocab_positions[:, -num_logits_to_keep:].unsqueeze(-1),
                float("-inf")
            )

        if self.expanded_vocab_size is not None:
            expanded_logits = self.expanded_lm_head(hidden_states[:, -num_logits_to_keep:, :])
            expanded_logits = expanded_logits.masked_fill(
                orig_vocab_positions[:, -num_logits_to_keep:].unsqueeze(-1),
                float("-inf")
            )
            logits = torch.cat([logits, expanded_logits], dim=-1)

        loss = None
        if labels is not None:
            if self.expanded_vocab_size is not None:
                _total_vocab_size = self.vocab_size + self.expanded_vocab_size
            else:
                _total_vocab_size = self.vocab_size

            loss = self.loss_function(logits=logits, labels=labels, vocab_size=_total_vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# NOTE(Shih-Lun): test script below
if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B"

    # Load GPT-2 model
    model = LlamaWithExpandedVocabForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        expanded_vocab_size=GPT2_VOCAB_SIZE,
        cache_dir=CACHE_DIR,
        torch_dtype="bfloat16",
    ).cuda()
    print("[INFO] Model loaded")
    print(model.expanded_vocab_size, model.model.expanded_vocab_size)
    model.eval()
    # exit()

    # Check consistency w/ llama original model
    llama_model = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype="bfloat16",
    ).cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        "/workspace/scratch-slseanwu/24-11-25-test-llama-finetune/llama-1b-test-finetune/checkpoint-200",
    )
    inputs = tokenizer([
        "Hello, my name is Julia, and today's weather is perfect",
        "I am a student at Stanford University, and I am studying computer science",
    ], 
    padding=True, truncation=True,
    return_tensors="pt"
    ).input_ids.cuda()

    new_outs = model(inputs, labels=inputs)
    orig_outs = llama_model(inputs, labels=inputs)
    assert torch.allclose(
        new_outs.logits[..., :llama_model.vocab_size],
        orig_outs.logits,
        atol=1e-4
    )

    print(new_outs.loss, orig_outs.loss)
    assert torch.allclose(new_outs.loss, orig_outs.loss, atol=1e-4)
    exit()

    # print(model.transformer.wpe.weight[:10, :4])
    # print(model.transformer.wpe.weight[:1024].std(), model.transformer.wpe.weight[1024:].std())

    # print trainable parameters count -- 782M
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # try some random input
    BSIZE = 4
    SEQLEN = 128
    COND_SEQLEN = 64
    VOCAB_SIZE = 10000
    llama_input_ids = torch.randint(0, VOCAB_SIZE, (BSIZE, COND_SEQLEN)).to(model.device)
    input_ids = torch.randint(0, VOCAB_SIZE, (BSIZE, SEQLEN)).to(model.device)

    outputs = model(input_ids, llama_input_ids=llama_input_ids, labels=input_ids)
    print(outputs.logits.size())
    print(outputs.loss)