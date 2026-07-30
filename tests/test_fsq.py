import torch

from fact.modules.fsq import FSQ


def test_codebook_size():
    fsq = FSQ([8, 8, 8, 4, 4])
    assert fsq.codebook_size == 8192
    assert fsq.num_dims == 5


def test_dual_view_shapes_and_ranges():
    fsq = FSQ([5, 5, 5])
    z = torch.randn(2, 7, 3) * 3
    out = fsq(z)
    assert out.continuous.shape == (2, 7, 3)
    assert out.quantized.shape == (2, 7, 3)
    assert out.indices.shape == (2, 7)
    # The bounded view can exceed [-1, 1] by ~eps (FSQ uses (1+eps) slack).
    assert out.continuous.abs().max() <= 1.0 + 0.01
    assert out.quantized.abs().max() <= 1.0 + 1e-5
    assert out.indices.min() >= 0
    assert out.indices.max() < fsq.codebook_size


def test_index_roundtrip():
    """indices_to_codes must exactly invert forward's index computation."""
    for levels in ([5, 5, 5], [8, 8, 4], [3, 2]):
        fsq = FSQ(levels)
        z = torch.randn(4, 50, len(levels)) * 4
        out = fsq(z)
        recon = fsq.indices_to_codes(out.indices)
        assert torch.allclose(recon, out.quantized, atol=1e-5), levels


def test_all_indices_reachable():
    fsq = FSQ([3, 3])
    z = torch.randn(1, 5000, 2) * 10
    seen = set(fsq(z).indices.flatten().tolist())
    assert seen == set(range(9))


def test_straight_through_gradient():
    fsq = FSQ([5, 5])
    z = torch.randn(2, 4, 2, requires_grad=True)
    out = fsq(z)
    out.quantized.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


def test_continuous_close_to_quantized():
    """The two views describe the same state: bounded distance on the lattice."""
    fsq = FSQ([9] * 4)
    z = torch.randn(2, 10, 4) * 3
    out = fsq(z)
    # Rounding moves each dim at most half a lattice cell (1/half_width in [-1,1] space).
    max_gap = (out.continuous - out.quantized).abs().max()
    assert max_gap <= 0.5 / 4 + 1e-5  # half_width = 4 for 9 levels
