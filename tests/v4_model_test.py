"""ChessModelV4 (SE-ResNet distillation net) sanity tests.

Covers: forward shapes, checkpoint round-trip through load_trained_model's
arch_version dispatch (bit-identical weights, v3 models unaffected),
deterministic eval, and a tiny-batch overfit through the exact training loss
path — the same bar the audit set for v3.
"""

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_network import (
    BOARD_CHANNELS_V3, BOARD_ENCODING_VERSION_V3, ChessModelV4,
    MOVE_VOCAB_SIZE_V3, get_encoding_spec, make_collate_policy_batch,
    mask_illegal_logits, policy_loss_for_targets, _NO_MOVE_HISTORY,
)


def test_forward_shapes():
    model = ChessModelV4(filters=64, blocks=2)
    model.eval()
    x = torch.randn(3, BOARD_CHANNELS_V3, 8, 8)
    with torch.no_grad():
        policy, value = model(x)
    assert policy.shape == (3, MOVE_VOCAB_SIZE_V3)
    assert value.shape == (3,)
    assert float(value.abs().max()) <= 1.0
    print('  v4 forward shapes OK')


def test_untrained_auxiliary_planes_are_ignored():
    """Four-field distillation FENs cannot supervise clock/repetition planes.

    Existing v4 weights must therefore be invariant to channels 17-19 when a
    live six-field FEN contains a nonzero halfmove clock or repetition state.
    """
    torch.manual_seed(7)
    model = ChessModelV4(filters=16, blocks=1).eval()
    base = torch.randn(2, BOARD_CHANNELS_V3, 8, 8)
    changed = base.clone()
    changed[:, 17:, :, :] = torch.randn_like(changed[:, 17:, :, :]) * 100
    with torch.no_grad():
        policy_a, value_a = model(base)
        policy_b, value_b = model(changed)
    assert torch.equal(policy_a, policy_b)
    assert torch.equal(value_a, value_b)
    print('  v4 ignores unsupervised clock/repetition planes')


def test_checkpoint_roundtrip_dispatch():
    from load_model import load_trained_model

    model = ChessModelV4(filters=64, blocks=3)
    payload = {
        'model_state_dict': model.state_dict(),
        'board_encoding': BOARD_ENCODING_VERSION_V3,
        'arch_version': 'v4',
        'residual_filters': 64,
        'residual_blocks': 3,
        'epoch': 1,
        'val_loss': 9.9,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'v4_test.pt')
        torch.save(payload, path)
        loaded = load_trained_model(path)
    assert isinstance(loaded, ChessModelV4)
    assert loaded.filters == 64 and loaded.blocks == 3
    assert loaded.encoding_version == BOARD_ENCODING_VERSION_V3
    assert loaded.encoding_spec == get_encoding_spec(BOARD_ENCODING_VERSION_V3)
    src = model.state_dict()
    for name, tensor in loaded.state_dict().items():
        assert torch.equal(tensor.cpu(), src[name].cpu()), name
    x = torch.randn(2, BOARD_CHANNELS_V3, 8, 8,
                    device=next(loaded.parameters()).device)
    with torch.no_grad():
        p1, v1 = loaded(x)
        p2, v2 = loaded(x)
    assert torch.equal(p1, p2) and torch.equal(v1, v2)
    print('  v4 checkpoint dispatch + bit-identical weights + deterministic eval OK')


def test_v3_dispatch_unaffected():
    from load_model import load_trained_model
    from neural_network import ChessModelV3

    if not os.path.exists('model/grandmaster_resnet_v3.pt'):
        print('  SKIP v3 dispatch check: checkpoint missing')
        return
    loaded = load_trained_model('model/grandmaster_resnet_v3.pt')
    assert isinstance(loaded, ChessModelV3)
    print('  v3 checkpoints still dispatch to ChessModelV3')


def test_tiny_overfit():
    """256 real rows through the v4 net + production loss must memorize."""
    chunk = 'data/train_chunks_v3/chunk_0000.npz'
    if not os.path.exists(chunk):
        print('  SKIP overfit: v3 chunk missing')
        return
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    with np.load(chunk) as data:
        rows = rng.choice(len(data['move_idx']), size=256, replace=False)
        rows.sort()
        boards = data['boards'][rows]
        move_idx = data['move_idx'][rows]
        offsets = data['legal_move_offsets']
        legal = [data['legal_move_indices'][offsets[i]:offsets[i + 1]]
                 for i in rows]
        values = data['value_target'][rows]

    collate = make_collate_policy_batch(MOVE_VOCAB_SIZE_V3)
    samples = [
        (boards[i], _NO_MOVE_HISTORY, np.int64(move_idx[i]), legal[i],
         np.float32(1.0), np.float32(values[i]), np.int8(1))
        for i in range(256)
    ]
    batch = collate(samples)
    b, m, tgt, mask, w, val, tt = batch

    model = ChessModelV4(filters=64, blocks=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for step in range(300):
        policy, value = model(b)
        masked = mask_illegal_logits(policy, mask)
        loss = policy_loss_for_targets(masked, tgt, tt).mean() + \
            ((value - val) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        policy, value = model(b)
        masked = mask_illegal_logits(policy, mask)
        acc = float((masked.argmax(1) == tgt).float().mean())
        vmse = float(((value - val) ** 2).mean())
    assert acc >= 0.95, f'v4 tiny overfit failed: top1 {acc:.3f}'
    assert vmse < 0.05, f'v4 value head stuck: mse {vmse:.4f}'
    print(f'  v4 tiny overfit OK: top1={acc:.3f} value_mse={vmse:.4f}')


if __name__ == '__main__':
    test_forward_shapes()
    test_untrained_auxiliary_planes_are_ignored()
    test_checkpoint_roundtrip_dispatch()
    test_v3_dispatch_unaffected()
    test_tiny_overfit()
    print('v4 model tests passed')
