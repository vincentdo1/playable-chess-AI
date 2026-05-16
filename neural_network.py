"""Chess move prediction model — PyTorch, GPU-accelerated."""

import os
import glob
import re
import random

import numpy as np
import chess
import chess.pgn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Config
TRAIN_DIR  = os.environ.get('TRAIN_DIR', 'data/train_chunks')
VAL_DIR    = os.environ.get('VAL_DIR', 'data/val_chunks')
MODEL_PATH = os.environ.get('MODEL_PATH', 'model/grandmaster_model_policy_v1.pt')
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '512'))
EPOCHS     = int(os.environ.get('EPOCHS', '50'))
SEQ_LEN    = 10
LR         = float(os.environ.get('LR', '1e-3'))
TRAIN_NUM_WORKERS = int(os.environ.get('TRAIN_NUM_WORKERS', '4'))
VAL_NUM_WORKERS = int(os.environ.get('VAL_NUM_WORKERS', '2'))

def _optional_int_env(name):
    value = os.environ.get(name)
    return int(value) if value else None

MAX_TRAIN_BATCHES = _optional_int_env('MAX_TRAIN_BATCHES')
MAX_VAL_BATCHES = _optional_int_env('MAX_VAL_BATCHES')
SAVE_MODEL = os.environ.get('SAVE_MODEL', '1').lower() not in {
    '0', 'false', 'no'
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

piece_to_index = {
    'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
    'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11,
    'ck': 12, 'cq': 13, 'CK': 14, 'CQ': 15,
    'ep': 16,
}
BOARD_CHANNELS = len(piece_to_index)

PROMOTION_OPTIONS = (None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
PROMOTION_TO_INDEX = {promotion: i for i, promotion in enumerate(PROMOTION_OPTIONS)}
INDEX_TO_PROMOTION = {i: promotion for promotion, i in PROMOTION_TO_INDEX.items()}
MOVE_VOCAB_SIZE = 64 * 64 * len(PROMOTION_OPTIONS)

# Data helpers

def fen_to_tensor(fen, flip: bool = False):
    """FEN -> 8x8x17 tensor. flip=True rotates for Black so the model always sees its own pieces at the bottom."""
    tensor = np.zeros((8, 8, BOARD_CHANNELS), dtype=np.float32)
    parts = fen.split(' ')
    rows = parts[0].split('/')
    for i, row in enumerate(rows):
        tensor_row = 7 - i
        j = 0
        for char in row:
            if char.isdigit():
                j += int(char)
            else:
                tensor[tensor_row, j, piece_to_index[char]] = 1
                j += 1
    castling = parts[2]
    if 'K' in castling: tensor[:, :, piece_to_index['CK']] = 1
    if 'Q' in castling: tensor[:, :, piece_to_index['CQ']] = 1
    if 'k' in castling: tensor[:, :, piece_to_index['ck']] = 1
    if 'q' in castling: tensor[:, :, piece_to_index['cq']] = 1

    en_passant = parts[3] if len(parts) > 3 else '-'
    if en_passant != '-':
        ep_square = chess.parse_square(en_passant)
        ep_rank = chess.square_rank(ep_square)
        ep_file = chess.square_file(ep_square)
        tensor[ep_rank, ep_file, piece_to_index['ep']] = 1

    if flip:
        # Rotate board 180 degrees: reverse both rows and columns
        tensor = tensor[::-1, ::-1, :].copy()

    return tensor

def square_to_index(square):
    return chess.parse_square(square.lower()[:2])

def flip_square(sq: int) -> int:
    """Mirror a square index 180 degrees (a1=0 <-> h8=63)."""
    return 63 - sq

def move_to_policy_index(move: chess.Move, flip: bool = False) -> int:
    """Encode a move as one fixed policy class: (from, to, promotion)."""
    from_sq = flip_square(move.from_square) if flip else move.from_square
    to_sq = flip_square(move.to_square) if flip else move.to_square
    try:
        promo_idx = PROMOTION_TO_INDEX[move.promotion]
    except KeyError as exc:
        raise ValueError(f"Unsupported promotion piece: {move.promotion}") from exc
    return ((from_sq * 64) + to_sq) * len(PROMOTION_OPTIONS) + promo_idx

def policy_index_to_move(index: int, flip: bool = False) -> chess.Move:
    """Decode a fixed policy class back to a chess.Move."""
    if not 0 <= index < MOVE_VOCAB_SIZE:
        raise ValueError(f"Policy index out of range: {index}")
    move_code, promo_idx = divmod(index, len(PROMOTION_OPTIONS))
    from_sq, to_sq = divmod(move_code, 64)
    if flip:
        from_sq = flip_square(from_sq)
        to_sq = flip_square(to_sq)
    return chess.Move(from_sq, to_sq, promotion=INDEX_TO_PROMOTION[promo_idx])

def legal_policy_indices(board: chess.Board, flip: bool = False):
    return np.array(
        [move_to_policy_index(move, flip=flip) for move in board.legal_moves],
        dtype=np.int32,
    )

def move_to_vector(move, flip: bool = False):
    """Encode a chess.Move as a 132-dim float vector."""
    vector = np.zeros(132, dtype=np.float32)
    from_sq = flip_square(move.from_square) if flip else move.from_square
    to_sq   = flip_square(move.to_square)   if flip else move.to_square
    vector[from_sq] = 1
    vector[64 + to_sq] = 1
    if move.promotion is not None:
        promo_map = {chess.KNIGHT: 128, chess.BISHOP: 129,
                     chess.ROOK: 130,  chess.QUEEN: 131}
        idx = promo_map.get(move.promotion)
        if idx is not None:
            vector[idx] = 1
    return vector

def move_sequence_to_vector(move_sequence, max_length=10, flip: bool = False):
    """Encode a list of chess.Move objects as a (max_length, 132) matrix."""
    seq = np.zeros((max_length, 132), dtype=np.float32)
    for i, move in enumerate(move_sequence[-max_length:]):
        seq[i] = move_to_vector(move, flip=flip)
    return seq

def move_to_index(move, board):
    """Return (from_sq, to_sq) as integers. Flipping is handled by the caller."""
    if move not in board.legal_moves:
        raise ValueError(f"Move {move.uci()} is not legal in this position.")
    return move.from_square, move.to_square

# Dataset

class ChunkDataset(torch.utils.data.IterableDataset):
    """Streams .npz chunk files sequentially — no random-access disk thrashing."""
    def __init__(self, chunk_dir, shuffle=True):
        self.chunk_paths = sorted(glob.glob(os.path.join(chunk_dir, 'chunk_*.npz')))
        if not self.chunk_paths:
            raise FileNotFoundError(
                f"No chunk files found in {chunk_dir!r}. Run preprocess.py first."
            )
        self.shuffle = shuffle
        total = 0
        for path in self.chunk_paths:
            with np.load(path) as data:
                total += len(data['boards'])
        print(f"Found {len(self.chunk_paths)} chunks in {chunk_dir}  ({total:,} positions)")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        paths = self.chunk_paths.copy()
        if worker_info is not None:
            paths = paths[worker_info.id::worker_info.num_workers]
        if self.shuffle:
            random.shuffle(paths)

        for path in paths:
            with np.load(path) as data:
                boards  = data['boards']
                moves   = data['moves']
                move_idx = data['move_idx'] if 'move_idx' in data else None
                legal_move_indices = (
                    data['legal_move_indices'] if 'legal_move_indices' in data
                    else None
                )
                legal_move_offsets = (
                    data['legal_move_offsets'] if 'legal_move_offsets' in data
                    else None
                )
                sample_weight = data['sample_weight'] if 'sample_weight' in data else None

            if boards.shape[-1] != BOARD_CHANNELS:
                raise ValueError(
                    f"{path} has {boards.shape[-1]} board channels, but this "
                    f"model expects {BOARD_CHANNELS}. Re-run preprocess.py "
                    "after architecture changes."
                )
            if move_idx is None or legal_move_indices is None or legal_move_offsets is None:
                raise ValueError(
                    f"{path} is missing policy-head training fields. "
                    "Re-run preprocess.py after architecture changes."
                )
            indices = np.random.permutation(len(boards)) if self.shuffle else np.arange(len(boards))
            for i in indices:
                board = torch.tensor(boards[i],  dtype=torch.float32).permute(2, 0, 1)
                move  = torch.tensor(moves[i],   dtype=torch.float32)
                target = torch.tensor(move_idx[i], dtype=torch.long)
                start = legal_move_offsets[i]
                end = legal_move_offsets[i + 1]
                legal = torch.tensor(legal_move_indices[start:end], dtype=torch.long)
                weight_value = 1.0 if sample_weight is None else sample_weight[i]
                weight = torch.tensor(weight_value, dtype=torch.float32)
                yield board, move, target, legal, weight

def collate_policy_batch(batch):
    boards, moves, targets, legal_indices, weights = zip(*batch)
    legal_mask = torch.zeros((len(batch), MOVE_VOCAB_SIZE), dtype=torch.bool)
    for row, indices in enumerate(legal_indices):
        legal_mask[row, indices] = True
        if not legal_mask[row, targets[row]]:
            raise ValueError("Training target is missing from its legal move mask.")
    return (
        torch.stack(boards),
        torch.stack(moves),
        torch.stack(targets),
        legal_mask,
        torch.stack(weights),
    )

def mask_illegal_logits(policy_logits, legal_mask):
    """Mask illegal moves with a value that is safe for fp32 and fp16."""
    mask_value = torch.finfo(policy_logits.dtype).min
    return policy_logits.masked_fill(~legal_mask, mask_value)

# Model

class ChessModel(nn.Module):
    """CNN (board) + LSTM (move history) -> fixed legal-masked move policy."""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.lstm = nn.LSTM(input_size=132, hidden_size=64, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(1088, 256),
            nn.ReLU(),
            nn.Linear(256, 128),  # Dropout removed — hurts chess CNN performance
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, MOVE_VOCAB_SIZE)

    def forward(self, board, moves):
        cnn_out = self.cnn(board)
        _, (hidden, _) = self.lstm(moves)
        lstm_out = hidden.squeeze(0)
        combined = torch.cat([cnn_out, lstm_out], dim=1)
        z = self.fc(combined)
        return self.policy_head(z)

# Training

def train_one_epoch(model, loader, optimizer, scaler, criterion, device,
                    max_batches=None):
    model.train()
    total_loss = total_weight = correct = total = 0
    for batch_idx, (boards, moves, move_idx, legal_mask, sample_weight) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        boards = boards.to(device)
        moves = moves.to(device)
        move_idx = move_idx.to(device)
        legal_mask = legal_mask.to(device)
        sample_weight = sample_weight.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device.type, enabled=device.type == 'cuda'):
            policy_logits = model(boards, moves)
            masked_logits = mask_illegal_logits(policy_logits, legal_mask)
            loss_per_sample = criterion(masked_logits, move_idx)
            weighted_loss = loss_per_sample * sample_weight
            loss = weighted_loss.sum() / sample_weight.sum().clamp_min(1e-6)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        n = boards.size(0)
        total_loss += weighted_loss.sum().item()
        total_weight += sample_weight.sum().item()
        correct += (masked_logits.argmax(1) == move_idx).sum().item()
        total += n
    if total == 0:
        raise RuntimeError("No training batches were produced by the DataLoader.")
    return total_loss / total_weight, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss = total_weight = correct = total = 0
    for batch_idx, (boards, moves, move_idx, legal_mask, sample_weight) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        boards = boards.to(device)
        moves = moves.to(device)
        move_idx = move_idx.to(device)
        legal_mask = legal_mask.to(device)
        sample_weight = sample_weight.to(device)
        with torch.amp.autocast(device.type, enabled=device.type == 'cuda'):
            policy_logits = model(boards, moves)
            masked_logits = mask_illegal_logits(policy_logits, legal_mask)
            loss_per_sample = criterion(masked_logits, move_idx)
            weighted_loss = loss_per_sample * sample_weight
        n = boards.size(0)
        total_loss += weighted_loss.sum().item()
        total_weight += sample_weight.sum().item()
        correct += (masked_logits.argmax(1) == move_idx).sum().item()
        total += n
    if total == 0:
        raise RuntimeError("No validation batches were produced by the DataLoader.")
    return total_loss / total_weight, correct / total

def main():
    model_dir = os.path.dirname(MODEL_PATH)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    print("Loading training dataset...")
    train_ds = ChunkDataset(TRAIN_DIR, shuffle=True)
    print("Loading validation dataset...")
    val_ds   = ChunkDataset(VAL_DIR,   shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=TRAIN_NUM_WORKERS,
        pin_memory=DEVICE.type == 'cuda',
        persistent_workers=TRAIN_NUM_WORKERS > 0,
        collate_fn=collate_policy_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=VAL_NUM_WORKERS,
        pin_memory=DEVICE.type == 'cuda',
        persistent_workers=VAL_NUM_WORKERS > 0,
        collate_fn=collate_policy_batch,
    )

    model     = ChessModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(reduction='none')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda')

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size : {BATCH_SIZE}  |  Max epochs : {EPOCHS}\n")
    if MAX_TRAIN_BATCHES is not None or MAX_VAL_BATCHES is not None:
        print(f"Debug batch limits: train={MAX_TRAIN_BATCHES}  "
              f"val={MAX_VAL_BATCHES}\n")

    best_val_loss       = float('inf')
    patience_count      = 0
    EARLY_STOP_PATIENCE = 5

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, DEVICE,
            max_batches=MAX_TRAIN_BATCHES)
        vl_loss, vl_acc = evaluate(
            model, val_loader, criterion, DEVICE,
            max_batches=MAX_VAL_BATCHES)
        print(f"  train  loss={tr_loss:.4f}  move_acc={tr_acc:.4f}")
        print(f"  val    loss={vl_loss:.4f}  move_acc={vl_acc:.4f}")
        scheduler.step(vl_loss)
        if vl_loss < best_val_loss:
            best_val_loss  = vl_loss
            patience_count = 0
            if SAVE_MODEL:
                torch.save({
                    'epoch':                epoch,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss':             best_val_loss,
                }, MODEL_PATH)
                print(f"  Saved best model (val_loss={best_val_loss:.4f})")
            else:
                print(f"  SAVE_MODEL=0, skipped checkpoint "
                      f"(val_loss={best_val_loss:.4f})")
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{EARLY_STOP_PATIENCE})")
            if patience_count >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    if SAVE_MODEL:
        print(f"\nTraining complete. Best model saved to: {MODEL_PATH}")
    else:
        print("\nTraining complete. SAVE_MODEL=0, no checkpoint written.")

if __name__ == '__main__':
    main()
