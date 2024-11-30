import logging
from typing import Optional, Union, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    LlamaForCausalLM,
    LlamaModel,
    LlamaConfig,
    AutoTokenizer,
    Cache,
    DynamicCache,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    logger,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)


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
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            raise ValueError("You should specify `input_ids` for this model")

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            # get original vocabulary positions
            orig_vocab_positions = torch.lt(input_ids, self.vocab_size).to(
                device=input_ids.device
            )
            orig_vocab_positions.requires_grad_(False)

            # clone input_ids for orig vocab positions and replace the rest with padding
            orig_input_ids = input_ids.clone()
            orig_input_ids[~orig_vocab_positions] = LLAMA_PAD_ID
            orig_inputs_embeds = self.embed_tokens(orig_input_ids)
            # replace padded positions with zeros with masked fill
            orig_inputs_embeds = orig_inputs_embeds.masked_fill(
                ~orig_vocab_positions.unsqueeze(-1), 0.0
            )

            # clone input_ids for expanded vocab positions and subtract all positions by vocab_size
            expanded_input_ids = input_ids.clone()
            expanded_input_ids -= self.vocab_size

            # replace original vocab positions with padding
            expanded_input_ids[orig_vocab_positions] = self.expanded_embed_tokens.padding_idx
            expanded_inputs_embeds = self.expanded_embed_tokens(expanded_input_ids)
            expanded_inputs_embeds = self.expanded_embed_proj(expanded_inputs_embeds)
            # replace padded positions with zeros with masked fill
            expanded_inputs_embeds = expanded_inputs_embeds.masked_fill(
                orig_vocab_positions.unsqueeze(-1), 0.0
            )

            inputs_embeds = orig_inputs_embeds + expanded_inputs_embeds


        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


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
        "I am a student at Stanford University, and I am studying computer science",
        "Hello, today's weather seems nice, but I don't feel great",
    ], 
    padding=True, truncation=True,
    return_tensors="pt"
    ).input_ids.cuda()

    # extend inputs for new model with expanded vocab
    extended_inputs = torch.LongTensor([
        [55026, 0, 10001, 19382],
        [55026, 0, 10001, 19382],
    ]).cuda() + model.vocab_size
    concat_inputs = torch.cat([inputs, extended_inputs], dim=-1)

    # sanity check that things are consistent if there are only text tokens
    new_outs = model(concat_inputs, labels=concat_inputs)
    orig_outs = llama_model(inputs, labels=inputs)

    print(new_outs.logits[:, :orig_outs.logits.size(1) - 1, 200:300])
    print(orig_outs.logits[:, :-1, 200:300])

    # assert torch.allclose(
    #     new_outs.logits[:, :orig_outs.logits.size(1) - 1, :llama_model.vocab_size],
    #     orig_outs.logits[:, :-1],
    #     atol=1e-2,
    #     rtol=1e-2
    # )

    print(new_outs.loss, orig_outs.loss)
    # assert torch.allclose(new_outs.loss, orig_outs.loss, atol=1e-4)

    new_outs.loss.backward()

    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        num_return_sequences=1,
    )

    print("\n[TEST GENERATION 1]")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    print("\n[TEST GENERATION 2]")
    print(tokenizer.decode(outputs[1], skip_special_tokens=True))

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