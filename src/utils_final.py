import torch
import tiktoken
import json
import numpy as np
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from torch.nn import (
    Parameter,
    Module,
    Linear,
    ModuleList,
    Dropout,
    Embedding,
    Sequential,
)
from scale_dot_product_gpa import scaled_dot_product_gqa

import matplotlib.pyplot as plt
import torch
import matplotlib.colors as mcolors


def create_dataloader_v1(
    txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True
) -> DataLoader:
    # Get toknizer
    tokenizer = tiktoken.get_encoding("gpt2")
    # Load dataset, but slicing windows
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    # Return batches in tensors
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    return dataloader


class GPTModel(Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = Dropout(cfg["drop_rate"])
        self.trf_blocks = Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class TransformerBlock(Module):
    def __init__(self, cfg):
        super().__init__()

        if cfg["attention_type"] == "MHA" or cfg["attention_type"] == "":
            self.att = MultiHeadAttention(
                d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                context_length=cfg["context_length"],
                num_heads=cfg["n_heads"],
                dropout=cfg["drop_rate"],
                qkv_bias=cfg["qkv_bias"],
            )
        elif cfg["attention_type"] == "MHGQA":
            self.att = MultiheadGQA(d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                context_length=cfg["context_length"],
                num_heads=cfg["n_heads"],
                dropout=cfg["drop_rate"],
                qkv_bias=cfg["qkv_bias"],
                kv_heads=cfg["kv_heads"])
        
        
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut
        return x


class FeedForward(Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = Sequential(
            Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class GELU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class LayerNorm(Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = Parameter(torch.ones(emb_dim))
        self.shift = Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_len, stride) -> None:
        super().__init__()
        # Init tokneizer
        self.tokenizer = tokenizer

        # DS to save inputs an targes, to train a LLM
        self.inputs_ids = []
        self.targets_ids = []

        # Get tokens for all text
        tokens_id = tokenizer.encode(txt)

        for i in range(0, len(tokens_id) - max_len, stride):
            # Get window slicing input and target
            input_chunk = tokens_id[i : i + max_len]
            target_chunk = tokens_id[i + 1 : i + 1 + max_len]
            # save input and target
            self.inputs_ids.append(torch.tensor(input_chunk))
            self.targets_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.inputs_ids)

    def __getitem__(self, idx) -> tuple:
        return self.inputs_ids[idx], self.targets_ids[idx]



class MultiHeadAttention(Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # A
        self.W_query = Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = Linear(d_out, d_out)  # B
        self.dropout = Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        torch.triu(torch.ones(context_length, context_length), diagonal=1)
        # variables to save attention weights for visualization
        self.atten_weights = None
        self.context_vector = None

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        self.atten_weights = attn_weights
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        self.context_vector = context_vec
        return context_vec



from einops import rearrange, einsum

class MultiheadGQA(Module):
    """
    A drop-in replacement for the old `MultiHeadAttention` class, but uses
    Grouped-Query Attention (GQA) internally. It has the same constructor signature
    and forward usage (`forward(x)`) as the original code, so it can be used
    similarly in your model, yet relies on the GQA mechanism under the hood.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        kv_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        # For GQA, we choose fewer key/value heads; e.g. 1 => multi-query
        # or some factor < num_heads for generalized GQA.
        # Here, we pick 1 by default for demonstration (MQA style).
        self.kv_heads = kv_heads

        self.head_dim = d_out // num_heads
        kv_embed_dim = self.head_dim * self.kv_heads

        # Query projects to (d_out)
        self.W_query = Linear(d_in, d_out, bias=qkv_bias)
        # Key, Value project to a reduced dimension => fewer kv heads
        self.W_key   = Linear(d_in, kv_embed_dim, bias=qkv_bias)
        self.W_value = Linear(d_in, kv_embed_dim, bias=qkv_bias)

        # Final projection back to d_out
        self.out_proj = Linear(d_out, d_out, bias=qkv_bias)

        # We'll store the (upper-triangular) causal mask for a sequence of `context_length`.
        # The original code uses an upper-tri mask and then in forward() calls masked_fill_().
        # 1's in the upper-tri => masked out, so only the lower tri (past) is allowed.
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

        # We'll store dropout probability (the GQA function expects a float).
        self.dropout_p = dropout

        # (Optional) store attn weights & context for visualization
        self.atten_weights = None
        self.context_vector = None

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_in)

        Returns:
            Tensor of shape (batch, seq_len, d_out)
        """
        b, n, _ = x.shape

        # 1) Project input to Q/K/V
        q = self.W_query(x)   # (b, n, d_out)
        k = self.W_key(x)     # (b, n, kv_embed_dim)
        v = self.W_value(x)   # (b, n, kv_embed_dim)

        # 2) Reshape:
        #    Q -> (b, n, num_heads, head_dim)
        #    K,V -> (b, n, kv_heads, head_dim)
        q = rearrange(q, "b n (h d) -> b n h d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.kv_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.kv_heads)

        # 4) Call the GQA attention function
        out, attn_weights = scaled_dot_product_gqa(
            query=q,
            key=k,
            value=v,
            dropout=self.dropout_p if self.training else 0.0,
            is_causal=True,      # we already passed an explicit mask
            need_weights=True,   # so we can store them
            average_attn_weights=False,
            force_grouped=True,  # ensure the grouped path is used
        )

        out = out.transpose(1, 2)  # Now out is (b, n, num_heads, head_dim)
        context_vec = out.contiguous().view(b, n, self.d_out)

        # 7) Final linear projection.
        context_vec = self.out_proj(context_vec)
        self.atten_weights = attn_weights.permute(0, 3, 1, 2)
        self.context_vector = context_vec
        return context_vec


class AttentionVisualizer:
    """
    Utility class to visualize attention weights from a Transformer model.
    """

    def __init__(self, model, tokenizer):
        """
        Initialize the visualizer with a model and tokenizer.

        Args:
            model: The Transformer model.
            tokenizer: The tokenizer used for encoding input.
        """
        self.model = model
        self.tokenizer = tokenizer

    def infer_and_visualize_attention(
        self,
        input_text,
        heads,
        layers,
        save_path="attention_weights.png",
        use_power_scale=False,
        gamma=0.5,
    ):
        """
        Perform inference and visualize attention weights for given heads and layers.

        Args:
            input_text (str): The input text to analyze.
            heads (list[int]): List of attention heads to visualize.
            layers (list[int]): List of Transformer layers to analyze.
            save_path (str): Path to save the attention weights plot.
            use_power_scale (bool): Whether to apply power scale normalization.
            gamma (float): The gamma value for power normalization if use_power_scale is True.
        """
        # Encode the input text
        encoded_input = self.tokenizer.encode(input_text)
        encoded_input_tensor = torch.tensor(encoded_input).unsqueeze(0)
        print("Next tokens are calculated from the following input:", input_text)

        # Decode tokens for labels
        labels = [self.tokenizer.decode([_token]) for _token in encoded_input]
        print("Labels:", labels)
        # In the plot, set labels size on axis x , y and title, to 0.5
        plt.rc("xtick", labelsize=4)
        plt.rc("ytick", labelsize=4)
        plt.rc("axes", titlesize=4)

        # Set model to evaluation mode and run inference
        self.model.eval()
        self.model(encoded_input_tensor)

        # Create a multiplot figure
        fig, axes = plt.subplots(len(layers), len(heads), figsize=(8, 8))

        # Handle case where axes may not be a 2D array
        if len(layers) == 1 and len(heads) == 1:
            axes = [[axes]]
        elif len(layers) == 1:
            axes = [axes]
        elif len(heads) == 1:
            axes = [[ax] for ax in axes]

        for i, layer in enumerate(layers):
            for j, head in enumerate(heads):
                # Extract attention weights
                attention_weights = self.model.trf_blocks[layer].att.atten_weights
                attention_matrix = attention_weights[0][head].detach().numpy()

                # Plot attention weights for the specified head
                ax = axes[i][j]
                self._plot_attention(
                    ax, attention_matrix, labels, head, layer, use_power_scale, gamma
                )

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        print(f"Attention weights saved to {save_path}")

    def _plot_attention(
        self,
        ax,
        attention_matrix,
        labels,
        head,
        layer,
        use_power_scale,
        gamma,
    ):
        """
        Plot attention weights as a heatmap on a given axis.

        Args:
            ax: Matplotlib axis to plot on.
            attention_matrix (numpy.ndarray): The attention weights matrix.
            labels (list): Token labels for x and y axes.
            head (int): The attention head being visualized.
            layer (int): The Transformer layer being visualized.
            use_power_scale (bool): Whether to apply power scale normalization.
            gamma (float): Gamma value for power normalization.
        """
        # Select colormap and normalization
        cmap = "Blues"
        norm = (
            mcolors.PowerNorm(gamma=gamma, vmin=0, vmax=1)
            if use_power_scale
            else mcolors.Normalize(vmin=0, vmax=1)
        )

        # Plot attention weights
        im = ax.matshow(attention_matrix, cmap=cmap, norm=norm)

        # Add labels to axes
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)

        # Add title
        ax.set_title(f"Layer {layer}, Head {head}")

        # Add colorbar nd size
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # reset loss gradients from the previous batch
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # calculate loss gradients
            optimizer.step()  # update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )
        generate_and_print_sample(
            model, train_loader.dataset.tokenizer, device, start_context
        )
        # torch.save(model.state_dict(), "models/it-%s-gpu1.model" % global_step)
    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(
    model,
    tokenizer,
    device,
    start_context,
    max_new_tokens=10,
    plot_best_tokens_prob=False,
    print_output=True,
    proba_threshold=0.001,
    save_path=False,
):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            tokenizer=tokenizer,
            plot_best_tokens_prob=plot_best_tokens_prob,
            proba_threshold=proba_threshold,
            save_path=save_path,
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        if print_output:
            print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()
    return decoded_text


def generate(
    model, idx, max_new_tokens, context_size, tokenizer, temperature, top_k=None
):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    if num_batches == 0:
        return np.NaN
    return total_loss / num_batches


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    # check if there is a btach dimension before squeezing
    if len(token_ids.shape) == 2:
        flat = token_ids.squeeze(0)  # remove batch dimension
    else:
        flat = token_ids
    return tokenizer.decode(flat.tolist())


def generate_text_simple(
    model,
    idx,
    max_new_tokens,
    context_size,
    tokenizer,
    plot_best_tokens_prob=False,
    proba_threshold=0.001,
    save_path=False,
):
    model.eval()
    for _next_token_ahead in range(max_new_tokens):
        idx_cond = idx[
            :, -context_size:
        ]  # keep only the input as large as the context_size limit
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[
            :, -1, :
        ]  # get last token probability distribution over vocabuary
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
        # Debug info, plot best tokens probability, when
        # a threshold is set by hand.
        if plot_best_tokens_prob and max_new_tokens != 1:
            print(
                "Error: max_new_tokens must be set to 1 to visualize next token prediction."
            )
            plot_best_tokens_prob = False

        if plot_best_tokens_prob:
            # get tokens over a propability threshold
            best_tokens = torch.where(probas > proba_threshold)[1]
            # Get best tokens probabilities and labels
            best_probs = probas[0][best_tokens]
            labels = []
            for i in range(best_tokens.shape[0]):
                labels.append(
                    str(_next_token_ahead)
                    + "'"
                    + tokenizer.decode(best_tokens[i : i + 1].numpy())
                    + "'"
                )
            # plot tokens and probs as bars
            # before plot bars with labels and best_probs, sorte descending
            idx = np.argsort(best_probs.numpy())[::-1]
            best_probs = best_probs[idx.copy()]
            labels = [labels[i] for i in idx]
            fig1 = plt.gcf()
            plt.bar(labels, best_probs)
            plt.xticks(rotation=45)
            # Enhance the plot with better labels and title
            plt.title("Next Token Prediction Probability Distribution", fontsize=16)
            plt.xlabel("Tokens", fontsize=14)
            plt.ylabel("Probability", fontsize=14)
            plt.xticks(rotation=45)
            plt.tight_layout()
            # save plot if save_path is set
            if save_path:
                # Show and save plot
                fig1.savefig(save_path)
                print(f"Attention weights saved to {save_path}")
            plt.show()
    return idx


def generate(model, idx, max_new_tokens, context_size, temperature, top_k=None):
    for _ in range(max_new_tokens):  # A
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:  # B
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )
        if temperature > 0.0:  # C
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:  # D
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def flatten_list(deep_list: list[list[object]]) -> list[object]:
    return list(chain.from_iterable(deep_list))


def create_tuples(input_list):
    return [(input_list[i], input_list[i + 1]) for i in range(len(input_list) - 1)]


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twiny()  # A
    ax2.plot(tokens_seen, train_losses, alpha=0)  # B
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()


def load_config(config_file):
    """
    Load GPT configuration from a JSON file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: GPT configuration.
    """
    with open(config_file, "r") as f:
        config = json.load(f)
    return config
