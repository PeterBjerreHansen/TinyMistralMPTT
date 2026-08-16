import torch
import torch.nn.functional as F

from tiny_mistral.modeling import MistralRMSNorm, apply_rotary_pos_emb, MistralRotaryEmbedding


def test_rmsnorm_matches_formula():
    x = torch.randn(2, 3, 8)
    norm = MistralRMSNorm(8, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 8))
    expected = norm.weight * (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6))
    torch.testing.assert_close(norm(x), expected)


def test_rope_preserves_vector_norm_pairwise_transform():
    B, H, T, D = 2, 4, 9, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    pos = torch.arange(T)[None, :].expand(B, -1)
    rope = MistralRotaryEmbedding(D)
    cos, sin = rope(k, pos)
    qr, kr = apply_rotary_pos_emb(q, k, cos, sin)
    torch.testing.assert_close(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(kr.norm(dim=-1), k.norm(dim=-1), atol=1e-5, rtol=1e-5)
