import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
import tiktoken
import os
from peft import PeftModel, PeftConfig

class LayerNorm(nn.Module):
    """ A custom LayerNorm module. """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """ The Causal Self-Attention mechanism. """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """ The Feed-Forward Network part of the Transformer block. """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """ A single Transformer block. """
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True

class GPT(nn.Module):
    """ The full GPT model. """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.5, top_p=0.9, top_k=None, repetition_penalty=1.1):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(idx_cond[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty  # Avoid amplifying negative logits

            # Apply top_k sampling
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.full_like(logits, -float('Inf'))
                logits.scatter_(1, top_k_indices, top_k_logits)

            # Apply top_p sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            if idx_next == 50256:  # Stop at <|endoftext|>
                break
                
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class Chatbot:
    def __init__(self, 
                 base_model_path="/root/GenAI_projects/SLM_Scratch_chatbot/slm_backend/best_model_params.pt",
                 peft_model_path="/root/GenAI_projects/SLM_Scratch_chatbot/slm_backend/story_slm_instruct_tune_peft"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enc = tiktoken.get_encoding("gpt2")
        
        print("Loading base model...")
        # Load base model
        config = GPTConfig()
        base_model = GPT(config)
        base_model.load_state_dict(torch.load(base_model_path, map_location=self.device))
        base_model.to(self.device)
        
        print("Loading Peft model...")
        # Load Peft model
        self.model = PeftModel.from_pretrained(base_model, peft_model_path)
        self.model.eval()
        
        print("Model loaded successfully!")

    def generate_response(self, prompt_text: str, max_new_tokens: int = 150) -> str:
        formatted = f"Instruction: {prompt_text}\n\nResponse:"
        start_ids = self.enc.encode(formatted, allowed_special={"<|endoftext|>"})
        x = torch.tensor(start_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            y = self.model.generate(
                x, 
                max_new_tokens=max_new_tokens,
                temperature=0.6,
                top_k=30,
                repetition_penalty=1.5
            )
        
        full_output = y[0].tolist()
        response_ids = full_output[len(start_ids):]
        
        if 50256 in response_ids:
            response_ids = response_ids[:response_ids.index(50256)]
        
        response = self.enc.decode(response_ids).strip()
        return response

# Test the model
if __name__ == "__main__":
    bot = Chatbot()
    while True:
        user = input("\nYou: ")
        if user.lower() in ["quit", "exit", "bye"]:
            print("Goodbye! 👋")
            break
        story = bot.generate_response(user)
        print(f"\nStory:\n{story}\n")