import torch

from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


CACHE_DIR = "/workspace/.cache"
LLAMA_PAD_ID = 128009 # "content": "<|eot_id|>"
GPT2_PAD_ID = 55025   # `SEPARATOR` in anticipation


class GPT2WithLlamaConditioning(GPT2LMHeadModel):
    def __init__(self, config, llama_model_name, new_max_seqlen=2048):
        super().__init__(config)
    
        # Load Llama model
        self.llama = LlamaForCausalLM.from_pretrained(
            llama_model_name,
            torch_dtype="bfloat16",
            cache_dir=CACHE_DIR,
            attn_implementation="flash_attention_2",
        )
        self.llama.requires_grad_(False)  # Freeze Llama for now (optional)
        self.llama.model.embed_tokens.padding_idx = LLAMA_PAD_ID

        # Linear layer to project Llama's hidden states to GPT-2's embedding space
        self.llama_projection = nn.Linear(
            self.llama.config.hidden_size, config.n_embd, bias=False
        )
        # initialize this with all zeros
        # nn.init.zeros_(self.llama_projection.weight)

        self.transformer.wte.padding_idx = GPT2_PAD_ID
        self.new_max_seqlen = new_max_seqlen


    def extend_pos_emb(self):
        if not self.new_max_seqlen > self.config.n_positions:
            return
        
        # extend GPT-2's position embeddings
        orig_emb = self.transformer.wpe.weight
        self.transformer.wpe = nn.Embedding(self.new_max_seqlen, self.transformer.config.n_embd)
        self.transformer.wpe.weight.data[:orig_emb.size(0)] = orig_emb.data

        # move to model device and dtype
        self.transformer.wpe.to(device=self.device, dtype=orig_emb.dtype)

        # adjust variances -- tensor(0.0145, device='cuda:0', grad_fn=<StdBackward0>) tensor(0.9989, device='cuda:0', grad_fn=<StdBackward0>)
        self.transformer.wpe.weight.data[orig_emb.size(0):] /= 100
        assert self.transformer.wpe.weight.requires_grad

        self.config.n_positions = self.new_max_seqlen


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        llama_input_ids=None,
        llama_attention_mask=None,
        **kwargs
    ):
        # Get Llama's hidden states
        llama_outputs = self.llama(
            input_ids=llama_input_ids, attention_mask=llama_attention_mask, output_hidden_states=True
        )
        llama_hidden_states = llama_outputs.hidden_states[-1]  # Use last hidden state
        # print(llama_hidden_states.shape)
        projected_llama_hidden_states = self.llama_projection(llama_hidden_states)

        # Add Llama's states to GPT-2's input embeddings
        gpt2_inputs_embeds = self.transformer.wte(input_ids)

        # print(gpt2_inputs_embeds.shape, projected_llama_hidden_states.shape)

        cond_seqlen = projected_llama_hidden_states.shape[1]
        # cond_seqlen = 0
        gen_seqlen = gpt2_inputs_embeds.shape[1]

        concat_position_ids = torch.cat(
            [
                torch.zeros(cond_seqlen, dtype=torch.long, device=projected_llama_hidden_states.device),
                torch.arange(gen_seqlen, dtype=torch.long, device=projected_llama_hidden_states.device),
            ],
            dim=0
        )
        concat_position_ids = concat_position_ids.unsqueeze(0)

        # remove the positional embedding for the condition part
        projected_llama_hidden_states -= self.transformer.wpe(concat_position_ids[:, :cond_seqlen])

        concat_input_embeds = torch.cat([projected_llama_hidden_states, gpt2_inputs_embeds], dim=1)

        # Forward pass through GPT-2

        outputs = super().forward(
            inputs_embeds=concat_input_embeds,
            position_ids=concat_position_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        logits = outputs.logits

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., cond_seqlen:-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs.loss = loss


        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past=None, llama_hidden_states=None, **kwargs
    ):
        # Pass Llama's hidden states during generation
        inputs = super().prepare_inputs_for_generation(input_ids, past=past, **kwargs)
        inputs["llama_hidden_states"] = llama_hidden_states
        return inputs

    def generate(self, llama_input_ids, llama_attention_mask, **kwargs):
        # Compute Llama's hidden states once and pass them during generation
        llama_outputs = self.llama(
            input_ids=llama_input_ids,
            attention_mask=llama_attention_mask,
            output_hidden_states=True,
        )
        llama_hidden_states = self.llama_projection(llama_outputs.hidden_states[-1])

        # Call GPT-2's generate method with conditioning
        return super().generate(
            llama_hidden_states=llama_hidden_states, **kwargs
        )


# NOTE(Shih-Lun): test script below
if __name__ == "__main__":
    GPT2_MODEL_NAME = "stanford-crfm/music-large-800k"
    LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B"

    # Load GPT-2 model
    model = GPT2WithLlamaConditioning.from_pretrained(
        GPT2_MODEL_NAME,
        llama_model_name=LLAMA_MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype="bfloat16",
        new_max_seqlen=1024,
    ).cuda()
    # model.extend_pos_emb()

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