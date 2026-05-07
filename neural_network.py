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
TRAIN_DIR  = 'data/train_chunks'
VAL_DIR    = 'data/val_chunks'
#MODEL_PATH = 'data/grandmaster_model_v2.pt'
MODEL_PATH = 'model/grandmaster_model_v2.pt'
BATCH_SIZE = 512
EPOCHS     = 50
SEQ_LEN    = 10
LR         = 1e-3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

piece_to_index = {
    'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
    'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11,
    'ck': 12, 'cq': 13, 'CK': 14, 'CQ': 15
}

# Data helpers

def fen_to_tensor(fen, flip: bool = False):
    """FEN → 8x8x16 tensor. flip=True rotates for Black so the model always sees its own pieces at the bottom."""
    tensor = np.zeros((8, 8, 16), dtype=np.float32)
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

    if flip:
        # Rotate board 180 degrees: reverse both rows and columns
        tensor = tensor[::-1, ::-1, :].copy()

    return tensor

def square_to_index(square):
    return chess.parse_square(square.lower()[:2])

def flip_square(sq: int) -> int:
    """Mirror a square index 180 degrees (a1=0 <-> h8=63)."""
    return 63 - sq

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
        total = sum(len(np.load(p)['boards']) for p in self.chunk_paths)
        print(f"Found {len(self.chunk_paths)} chunks in {chunk_dir}  ({total:,} positions)")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        paths = self.chunk_paths.copy()
        if self.shuffle:
            random.shuffle(paths)
        if worker_info is not None:
            paths = paths[worker_info.id::worker_info.num_workers]

        for path in paths:
            data    = np.load(path)
            boards  = data['boards']
            moves   = data['moves']
            from_sq = data['from_sq']
            to_sq   = data['to_sq']
            indices = np.random.permutation(len(boards)) if self.shuffle else np.arange(len(boards))
            for i in indices:
                board = torch.tensor(boards[i],  dtype=torch.float32).permute(2, 0, 1)
                move  = torch.tensor(moves[i],   dtype=torch.float32)
                f_sq  = torch.tensor(from_sq[i], dtype=torch.long)
                t_sq  = torch.tensor(to_sq[i],   dtype=torch.long)
                yield board, move, f_sq, t_sq

# Model

class ChessModel(nn.Module):
    """CNN (board) + LSTM (move history) → two 64-class heads (from-square, to-square)."""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
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
        self.from_head = nn.Linear(128, 64)
        self.to_head   = nn.Linear(128, 64)

    def forward(self, board, moves):
        cnn_out = self.cnn(board)
        _, (hidden, _) = self.lstm(moves)
        lstm_out = hidden.squeeze(0)
        combined = torch.cat([cnn_out, lstm_out], dim=1)
        z = self.fc(combined)
        return self.from_head(z), self.to_head(z)

# Training

def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    total_loss = from_correct = to_correct = total = 0
    for boards, moves, from_sq, to_sq in loader:
        boards  = boards.to(device)
        moves   = moves.to(device)
        from_sq = from_sq.to(device)
        to_sq   = to_sq.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            from_logits, to_logits = model(boards, moves)
            loss = criterion(from_logits, from_sq) + criterion(to_logits, to_sq)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        n = boards.size(0)
        total_loss   += loss.item() * n
        from_correct += (from_logits.argmax(1) == from_sq).sum().item()
        to_correct   += (to_logits.argmax(1)   == to_sq).sum().item()
        total        += n
    return total_loss / total, from_correct / total, to_correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = from_correct = to_correct = total = 0
    for boards, moves, from_sq, to_sq in loader:
        boards  = boards.to(device)
        moves   = moves.to(device)
        from_sq = from_sq.to(device)
        to_sq   = to_sq.to(device)
        with torch.amp.autocast('cuda'):
            from_logits, to_logits = model(boards, moves)
            loss = criterion(from_logits, from_sq) + criterion(to_logits, to_sq)
        n = boards.size(0)
        total_loss   += loss.item() * n
        from_correct += (from_logits.argmax(1) == from_sq).sum().item()
        to_correct   += (to_logits.argmax(1)   == to_sq).sum().item()
        total        += n
    return total_loss / total, from_correct / total, to_correct / total

def main():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    print("Loading training dataset...")
    train_ds = ChunkDataset(TRAIN_DIR, shuffle=True)
    print("Loading validation dataset...")
    val_ds   = ChunkDataset(VAL_DIR,   shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              num_workers=2, pin_memory=True,
                              persistent_workers=True)

    model     = ChessModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    scaler = torch.amp.GradScaler('cuda')

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size : {BATCH_SIZE}  |  Max epochs : {EPOCHS}\n")

    best_val_loss       = float('inf')
    patience_count      = 0
    EARLY_STOP_PATIENCE = 5

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        tr_loss, tr_from, tr_to = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, DEVICE)
        vl_loss, vl_from, vl_to = evaluate(
            model, val_loader, criterion, DEVICE)
        print(f"  train  loss={tr_loss:.4f}  from_acc={tr_from:.4f}  to_acc={tr_to:.4f}")
        print(f"  val    loss={vl_loss:.4f}  from_acc={vl_from:.4f}  to_acc={vl_to:.4f}")
        scheduler.step(vl_loss)
        if vl_loss < best_val_loss:
            best_val_loss  = vl_loss
            patience_count = 0
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':             best_val_loss,
            }, MODEL_PATH)
            print(f"  Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{EARLY_STOP_PATIENCE})")
            if patience_count >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nTraining complete. Best model saved to: {MODEL_PATH}")

if __name__ == '__main__':
    main()