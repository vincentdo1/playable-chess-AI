"""Exercise the production loss path with a tiny-batch overfit test.

The test checks legal-move masking at initialization and memorization after
training. It re-executes with a small CPU model configured before import.
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHILD_FLAG = 'AUDIT_OVERFIT_CHILD'

N_ROWS = 256
BATCH_SIZE = 128
STEPS = 400
LR = 1e-3
VALUE_LOSS_WEIGHT = 0.25


def run_child():
    sys.path.insert(0, REPO_ROOT)
    import numpy as np
    import torch

    from neural_network import (
        BOARD_ENCODING_VERSION_V3, ChessModelV3, get_encoding_spec,
        make_collate_policy_batch, mask_illegal_logits,
        policy_loss_for_targets,
    )

    assert os.environ.get('RESIDUAL_FILTERS') == '64'
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    with np.load(os.path.join(REPO_ROOT,
                              'data/train_chunks_v3/chunk_0000.npz')) as data:
        rows = rng.choice(len(data['move_idx']), size=N_ROWS, replace=False)
        rows.sort()
        boards = data['boards'][rows]
        move_idx = data['move_idx'][rows]
        offsets = data['legal_move_offsets']
        legal = [
            data['legal_move_indices'][offsets[i]:offsets[i + 1]]
            for i in rows
        ]
        value_target = data['value_target'][rows]

    spec = get_encoding_spec(BOARD_ENCODING_VERSION_V3)
    collate = make_collate_policy_batch(spec['move_vocab_size'])
    no_history = np.zeros((0,), dtype=np.float32)
    samples = [
        (boards[i], no_history, np.int64(move_idx[i]), legal[i],
         np.float32(1.0), np.float32(value_target[i]), np.int8(1))
        for i in range(N_ROWS)
    ]

    model = ChessModelV3()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def batch_loss(batch, train):
        b, m, tgt, mask, w, val, tt = batch
        model.train(train)
        policy_logits, value_pred = model(b, m)
        masked = mask_illegal_logits(policy_logits, mask)
        policy_loss = policy_loss_for_targets(masked, tgt, tt).mean()
        value_loss = ((value_pred.float() - val.float()) ** 2).mean()
        acc = (masked.argmax(1) == tgt).float().mean()
        return policy_loss, value_loss, acc

    # Initial loss should match the uniform distribution over legal moves.
    full_batch = collate(samples)
    with torch.no_grad():
        policy0, value0, acc0 = batch_loss(full_batch, train=False)
    uniform_entropy = float(np.mean([np.log(len(l)) for l in legal]))
    print(f'  initial policy loss {policy0:.3f} vs uniform-over-legal '
          f'{uniform_entropy:.3f} (ln vocab would be '
          f'{np.log(spec["move_vocab_size"]):.3f})')
    assert abs(float(policy0) - uniform_entropy) < 0.6, (
        'initial masked loss is far from the uniform-over-legal baseline — '
        'the legality mask is probably not being applied'
    )

    order = np.arange(N_ROWS)
    for step in range(1, STEPS + 1):
        rng.shuffle(order)
        for lo in range(0, N_ROWS, BATCH_SIZE):
            chunk = [samples[i] for i in order[lo:lo + BATCH_SIZE]]
            batch = collate(chunk)
            policy_loss, value_loss, _ = batch_loss(batch, train=True)
            loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                p, v, a = batch_loss(full_batch, train=False)
            print(f'  step {step:4d}: policy={float(p):.4f} '
                  f'value_mse={float(v):.4f} top1={float(a):.3f}')

    with torch.no_grad():
        p, v, a = batch_loss(full_batch, train=False)
    assert float(a) >= 0.95, (
        f'tiny-batch overfit failed: top-1 {float(a):.3f} < 0.95 — labels, '
        'masking, or gradient flow are broken'
    )
    assert float(p) < 0.35, f'policy loss stuck at {float(p):.3f}'
    assert float(v) < 0.05, f'value MSE stuck at {float(v):.3f}'
    print(f'  overfit OK: top1={float(a):.3f} policy={float(p):.4f} '
          f'value_mse={float(v):.4f} on {N_ROWS} rows')
    print('audit tiny-batch overfit test passed')


def main():
    if os.environ.get(CHILD_FLAG) == '1':
        run_child()
        return
    env = dict(os.environ)
    env.update({
        CHILD_FLAG: '1',
        'RESIDUAL_FILTERS': '64',
        'RESIDUAL_BLOCKS': '2',
        'TORCH_DEVICE': 'cpu',      # leave the GPU alone
        'TRAIN_NUM_WORKERS': '0',
        'VAL_NUM_WORKERS': '0',
    })
    result = subprocess.run([sys.executable, os.path.abspath(__file__)],
                            env=env, cwd=REPO_ROOT)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
