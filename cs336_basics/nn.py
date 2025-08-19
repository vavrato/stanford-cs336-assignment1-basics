import random
from typing import Optional
import numpy as np
from torch import nn, Tensor
import torch
import math
from einops import rearrange, reduce, repeat
from jaxtyping import Float, Int


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,  # type: ignore
        dtype: torch.dtype | None = None,  # type: ignore
    ) -> None:
        super().__init__()
        self.in_features = in_features

        var = 2 / (in_features + out_features)
        sigma = math.sqrt(var)

        W = torch.empty(out_features, in_features, device=device, dtype=dtype)
        W = nn.init.trunc_normal_(W, 0, sigma, a=-3 * sigma, b=3 * sigma)
        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,  # type: ignore
        dtype: torch.dtype | None = None,  # type: ignore
    ) -> None:
        super().__init__()
        W = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        W = nn.init.trunc_normal_(W, 0, 1, a=-3, b=3)
        self.W = nn.Parameter(W)

    def forward(self, x: Int[Tensor, "... vocab_size"]) -> torch.Tensor:
        return (self.W)[x]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,  # type: ignore
        dtype: torch.dtype | None = None,  # type: ignore
    ) -> None:
        super().__init__()
        self.eps = eps
        self.d_model = d_model

        self.W = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        x = x.to(torch.float32)

        RMS = reduce(x**2, "... d -> ... 1", "mean")
        RMS = torch.sqrt(RMS + self.eps)

        x = (x / RMS) * self.W

        return x.to(in_dtype)


class Silu(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)


def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)


class FFN_SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        device: torch.device | None = None,  # type: ignore
        dtype: torch.dtype | None = None,  # type: ignore
    ):
        super().__init__()
        if not d_ff:
            d_ff = int(d_model * 8 / 3)

        var_13 = 2 / (d_model + d_ff)
        sigma = math.sqrt(var_13)

        W1 = torch.empty(d_ff, d_model, device=device, dtype=dtype)
        W3 = torch.empty(d_ff, d_model, device=device, dtype=dtype)
        W1 = nn.init.trunc_normal_(W1, 0, sigma, a=-3 * sigma, b=3 * sigma)
        W3 = nn.init.trunc_normal_(W3, 0, sigma, a=-3 * sigma, b=3 * sigma)
        self.W1 = nn.Parameter(W1)
        self.W3 = nn.Parameter(W3)

        W2 = torch.empty(d_model, d_ff, device=device, dtype=dtype)
        W2 = nn.init.trunc_normal_(W2, 0, sigma, a=-3 * sigma, b=3 * sigma)
        self.W2 = nn.Parameter(W2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = silu(x @ self.W1.T)
        b = x @ self.W3.T

        return (a * b) @ self.W2.T


def ffn_swiglu(x: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor, W3: torch.Tensor) -> torch.Tensor:
    a = silu(x @ W1.T)
    b = x @ W3.T

    return (a * b) @ W2.T


class ROPE(nn.Module):
    # it would be nice to refactor, clean, and make more efficient
    # no time, though, two kids and fulltime job
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: Optional[torch.device] = None) -> None:
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.theta_base = theta ** (-2 / d_k)
        t = self.make_angle_tensor(max_seq_len, d_k)

        cos = torch.cos(t)
        sin = torch.sin(t)

        self.register_buffer(
            "cos", cos, persistent=False
        )  # persistent=False, otherwise problems with load_state_dict when this is a submodule
        self.register_buffer("sin", sin, persistent=False)

    def theta_i(self, i):
        return self.theta_base ** (i)

    def make_angle_vector(self, d_k) -> torch.Tensor:
        vector_list = []
        for k in range(0, d_k // 2):
            vector_list.append(self.theta_i(k))

        vector = torch.tensor(vector_list).reshape(1, -1)
        return repeat(vector, "a b -> a (b n)", n=2)

    def make_angle_tensor(self, max_seq_len, d_k) -> torch.Tensor:
        rows = []
        angle_vector = self.make_angle_vector(d_k)
        for m in range(0, max_seq_len):
            rows.append(m * angle_vector)

        return torch.cat(rows)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor]) -> torch.Tensor:
        # first make the rearranged vector that will be multiplied by sin, I don't know if this can be more elegant
        y = x.clone()

        y[..., 0::2] = -x[..., 1::2]
        y[..., 1::2] = x[..., 0::2]

        if positions is None:
            positions = torch.arange(x.shape[-2])
        cos = getattr(self, "cos")[positions, :]
        sin = getattr(self, "sin")[positions, :]

        return x * cos + y * sin


def softmax(x: torch.Tensor, *, dim: Optional[int] = None) -> torch.Tensor:
    x_shifted = x - torch.max(x, dim, keepdim=True).values
    exp = torch.exp(x_shifted)
    norm = exp.sum(dim, keepdim=True)

    return exp / norm


def sdpa(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Optional[Float[Tensor, " ... queries keys"]] = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    presoftmax = 1 / d_k ** (0.5) * (Q @ rearrange(K, "... keys d_model -> ... d_model keys"))

    if mask is not None:
        presoftmax += torch.where(mask, 0, -torch.inf)

    return softmax(presoftmax, dim=-1) @ V


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        max_seq_len: int = 1,
        use_rope: bool = False,
        theta: float = 10000,
    ):
        super().__init__()
        self.num_heads = num_heads
        if not d_k:
            d_k = int(d_model / num_heads)
        if not d_v:
            d_v = int(d_model / num_heads)

        sigma_qk = math.sqrt(2 / (d_model + d_k))
        sigma_v = math.sqrt(2 / (d_model + d_v))
        # I think this should be the correct init. The variance corresponds to the matrix corresponding to one head only. Here they are concated for efficiency
        self.WQ = Linear(d_k * num_heads, d_model).W
        nn.init.normal_(self.WQ, 0, sigma_qk)

        self.WK = Linear(d_k * num_heads, d_model).W
        nn.init.normal_(self.WK, 0, sigma_qk)

        self.WV = Linear(d_v * num_heads, d_model).W
        nn.init.normal_(self.WV, 0, sigma_v)

        self.WO = Linear(d_model, num_heads * d_v).W
        nn.init.normal_(self.WO, 0, sigma_v)

        self.use_rope = use_rope
        if use_rope:
            self.rope = ROPE(theta, d_k, max_seq_len)

    def forward(self, x: Float[Tensor, "... seq_len d_model"], token_positions=None) -> torch.Tensor:
        # this does just one matrix multiplication
        W = torch.cat(
            [self.WQ, self.WK, self.WV], dim=0
        )  # each W{Q,K,V} is (h*d_k, d_model), so we make (3 * h * d_k, d_model)

        QKV = x @ W.T  # this is ... seq_len 3*h*d_k
        QKV = rearrange(QKV, "... s (three h d_k) -> ... h s (three d_k)", h=self.num_heads, three=3)
        Q, K, V = QKV.chunk(3, dim=-1)  # each is now [... h s d]

        if self.use_rope:
            Q = self.rope.forward(Q, positions=token_positions)
            K = self.rope.forward(K, positions=token_positions)

        seq_len = x.shape[-2]
        mask = rearrange(torch.tril(torch.ones(seq_len, seq_len)).bool(), "s1 s2 -> 1 1 s1 s2")  # causal mask
        attention = sdpa(Q, K, V, mask=mask)  # batch, num_heads, seq_len, d_k
        attention = rearrange(attention, "b h s d -> b s (h d)")  # this is the concatenation

        # oof, this was hard

        return attention @ self.WO.T


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, max_seq_len=max_seq_len, theta=theta, use_rope=True)
        self.norm2 = RMSNorm(d_model)
        self.swiglu = FFN_SwiGLU(d_model, d_ff)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        x_res = x
        x = self.norm1.forward(x)
        x = self.mha.forward(x)
        x = x_res + x

        x_res = x
        x = self.norm2.forward(x)
        x = self.swiglu.forward(x)

        return x_res + x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = Embedding(vocab_size, d_model)
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, num_heads, d_ff, context_length, theta) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.linear(x)

        return x


def cross_entropy(logits: Float[Tensor, "b vocab_size"], targets: Int[Tensor, "b"]) -> Float[Tensor, ""]:
    first = logits[torch.arange(targets.shape[0]), targets]
    maxes_ = torch.max(logits, dim=-1, keepdim=True).values
    exps_ = torch.exp(logits - maxes_)
    second = (
        torch.log(torch.sum(exps_, dim=-1, keepdim=True)) + maxes_
    )  # or we can do keepdim=False with maxes_.squeeze()
    return -(first - second).mean()


class AdamW(torch.optim.Optimizer):
    # I'm starting being lazy with the type hints
    def __init__(self, params, lr=0.001, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                state["m"] = beta1 * state.get("m", torch.zeros_like(p)) + (1 - beta1) * grad
                state["v"] = beta2 * state.get("v", torch.zeros_like(p)) + (1 - beta2) * (grad**2)

                m = state["m"]
                v = state["v"]
                t = state.get("t", 1)

                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data *= 1 - lr * weight_decay

                state["t"] = t + 1

        return loss


def cosine_annealing(t: int, lr_max: float, lr_min: float, T_w: int, T_c: int) -> float:
    if t <= T_w:
        return lr_max * t / T_w
    if t <= T_c:
        return lr_min + 1 / 2 * (1 + math.cos((t - T_w) / (T_w - T_c) * math.pi)) * (lr_max - lr_min)
    return lr_min


def gradient_clipping(params, M: float):
    norm = torch.tensor(0.0)
    for p in params:
        if p.grad is None:
            continue
        norm += torch.sum(torch.square(p.grad.data))

    norm = torch.sqrt(norm)

    if norm > M:
        scale = M / (norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(scale)


def data_loader(
    dataset: np.ndarray, batch_size: int, context_length: int, device: Optional[str] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    indices = indices.reshape(-1, 1) + np.arange(context_length + 1)

    t = torch.from_numpy(dataset[indices]).to(dtype=torch.int64, device=device)

    return (t[:, :-1], t[:, 1:])


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}, out)


def load_checkpoint(src: str, model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    d = torch.load(src)
    model.load_state_dict(d["model"])
    optimizer.load_state_dict(d["optimizer"])

    return d["iteration"]
