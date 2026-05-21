"""Synapse-style visualization of the chess CNN+LSTM forward pass.

Runs a real chess position through the trained ChessModel, captures
activations at every layer via PyTorch hooks, and animates a "pulse"
of light sweeping left-to-right through the network. Layers are drawn
as columns of glowing nodes connected by faint synapses; node brightness
modulates with real activation magnitude.

The script is architecture-agnostic: it introspects the model at runtime,
so changing RESIDUAL_FILTERS or RESIDUAL_BLOCKS auto-rescales the output.

Usage:
    python scripts/visualize_network.py
    python scripts/visualize_network.py --fen <FEN> --frames 180
    python scripts/visualize_network.py --layout portfolio --output-name chess-network
    python scripts/visualize_network.py --no-mp4 --no-webm --frames 60
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
from matplotlib.collections import LineCollection
from PIL import Image

# Match the playable-chess-AI web UI design language (frontend/src/styles/app.css):
# deep navy + gold accent + warm cream text, dark-academia feel.
matplotlib.rcParams['font.serif'] = [
    'Cormorant Garamond', 'EB Garamond', 'Playfair Display',
    'Garamond', 'Times New Roman', 'DejaVu Serif',
]
matplotlib.rcParams['font.sans-serif'] = [
    'DM Sans', 'Inter', 'Helvetica', 'Arial', 'DejaVu Sans',
]

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import chess  # noqa: E402
from neural_network import (  # noqa: E402
    ChessModel, ResidualBlock, RESIDUAL_FILTERS, RESIDUAL_BLOCKS,
    policy_index_to_move,
)
from load_model import load_trained_model, _position_tensors  # noqa: E402


# Palette definitions. Switch at the CLI with --palette.
# Each palette has the full color set so the script stays neutral about
# which one is "default" - swap the active one via PALETTES[args.palette].
PALETTES: dict[str, dict] = {
    # Matches the playable-chess-AI web UI (frontend/src/styles/app.css):
    # deep navy + gold accent + warm cream, dark-academia feel.
    'gold': {
        'bg':           '#070C18',
        'title':        '#EDE5D0',
        'subtle':       '#8A8070',
        'faint':        '#4A4438',
        'cmap_stops':   [
            (0.00, '#2A2218'),
            (0.30, '#5C4A20'),
            (0.55, '#C9A43C'),
            (0.80, '#E2C060'),
            (1.00, '#F5E5A8'),
        ],
        'edge_rgb':     (0.79, 0.64, 0.24),
        'spark_rgb':    (0.96, 0.90, 0.66),
        'board_border': '#C9A43C',
        'demo_color':   '#7a5a32',
    },
    # Diverging cool-to-hot scheme: blue idle -> purple mid -> red peak.
    # "Energy transition" feel; reads as scientific / neural.
    'redblue': {
        'bg':           '#06080F',
        'title':        '#E8ECF5',
        'subtle':       '#7A8499',
        'faint':        '#3A4258',
        'cmap_stops':   [
            (0.00, '#0E1830'),
            (0.30, '#2C4882'),
            (0.55, '#7B4E96'),
            (0.80, '#E0556E'),
            (1.00, '#FF8E78'),
        ],
        'edge_rgb':     (0.48, 0.42, 0.72),
        'spark_rgb':    (1.00, 0.92, 0.92),
        'board_border': '#4A6FB0',
        'demo_color':   '#8A6B7A',
        'title_family': 'serif',
        'title_style':  'italic',
        'title_weight': 'light',
        'title_caps':   False,
    },
    # Phosphor-terminal aesthetic to match the vmd306.com portfolio:
    # pure black bg, neon green accent, white text, monospace uppercase.
    'terminal': {
        'bg':           '#000000',
        'title':        '#FFFFFF',
        'subtle':       '#A8A8A8',
        'faint':        '#4A4A4A',
        'cmap_stops':   [
            (0.00, '#0A1F0A'),
            (0.30, '#1A5A1A'),
            (0.55, '#36D936'),
            (0.80, '#74FF74'),
            (1.00, '#CCFFCC'),
        ],
        'edge_rgb':     (0.21, 0.85, 0.21),
        'spark_rgb':    (0.96, 1.00, 0.96),
        'board_border': '#36D936',
        'demo_color':   '#558855',
        'title_family': 'monospace',
        'title_style':  'normal',
        'title_weight': 'bold',
        'title_caps':   True,
    },
}

# The 'gold' palette is the default if a palette doesn't specify these
# (keeps backward compatibility for any external configs).
PALETTES['gold'].setdefault('title_family', 'serif')
PALETTES['gold'].setdefault('title_style',  'italic')
PALETTES['gold'].setdefault('title_weight', 'light')
PALETTES['gold'].setdefault('title_caps',   False)


def _resolve_palette(name: str):
    """Return the full visual config for the named palette."""
    p = PALETTES[name]
    cmap = LinearSegmentedColormap.from_list(f'chess_{name}', p['cmap_stops'])
    return {
        'bg':           p['bg'],
        'title':        p['title'],
        'subtle':       p['subtle'],
        'faint':        p['faint'],
        'cmap':         cmap,
        'edge_rgb':     p['edge_rgb'],
        'spark_rgb':    p['spark_rgb'],
        'board_border': p['board_border'],
        'demo_color':   p['demo_color'],
        'title_family': p.get('title_family', 'serif'),
        'title_style':  p.get('title_style',  'italic'),
        'title_weight': p.get('title_weight', 'light'),
        'title_caps':   p.get('title_caps',   False),
    }


# Defaults - overridden in main() once we know the palette choice.
BG_COLOR, TITLE_COLOR, SUBTLE_COLOR, FAINT_COLOR = '#070C18', '#EDE5D0', '#8A8070', '#4A4438'
NODE_CMAP = LinearSegmentedColormap.from_list('chess_default', PALETTES['gold']['cmap_stops'])
EDGE_COLOR  = PALETTES['gold']['edge_rgb']
SPARK_COLOR = PALETTES['gold']['spark_rgb']
BOARD_BORDER = PALETTES['gold']['board_border']
DEMO_COLOR   = PALETTES['gold']['demo_color']
TITLE_FAMILY = 'serif'
TITLE_STYLE  = 'italic'
TITLE_WEIGHT = 'light'
TITLE_CAPS   = False

# Display caps so wide layers stay legible
MAX_NODES_PER_COLUMN = 20
MAX_SYNAPSES_PER_GAP = 180
SPARKS_PER_GAP       = 4

# Pulse envelope half-width on the [0, 1] x-axis
PULSE_SIGMA   = 0.07
IDLE_FLOOR    = 0.15

RNG_SEED = 42


# ---------- Layer specs ----------

@dataclass
class LayerSpec:
    """One column in the visualization."""
    name: str
    short_label: str
    activations: np.ndarray            # 1D, length = display_n
    x: float = 0.0
    y_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    node_sizes: np.ndarray = field(default_factory=lambda: np.array([]))
    color: str = '#7aa7ff'             # used for short_label text


@dataclass(frozen=True)
class LayoutConfig:
    """Composition presets for different embed targets."""
    name: str
    show_title: bool = True
    show_board: bool = True
    show_readout: bool = True
    layer_x_left: float = 0.23
    layer_x_right: float = 0.94
    layer_y_low: float = 0.24
    layer_y_high: float = 0.82
    board_x0: float = 0.035
    board_x1: float = 0.175
    board_y_top: float = 0.84


LAYOUTS: dict[str, LayoutConfig] = {
    'full': LayoutConfig(name='full'),
    # Portfolio cards add their own title, badges, and gradient treatment.
    # Keep this export focused on the moving network so it does not fight
    # the card chrome or make the chess board look cramped.
    'card': LayoutConfig(
        name='card',
        show_title=False,
        show_board=False,
        show_readout=False,
        layer_x_left=0.055,
        layer_x_right=0.965,
        layer_y_low=0.18,
        layer_y_high=0.82,
    ),
    # Tailored for vincent-portfolio's featured project card:
    # top corners have metadata labels and the bottom band carries the
    # large project title, so keep the network in the card's open field.
    'portfolio': LayoutConfig(
        name='portfolio',
        show_title=False,
        show_board=False,
        show_readout=False,
        layer_x_left=0.24,
        layer_x_right=0.965,
        layer_y_low=0.31,
        layer_y_high=0.82,
    ),
}


# ---------- Model loading ----------

def load_model_or_random():
    """Return (model, used_random_init)."""
    try:
        model = load_trained_model()
        return model, False
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        warnings.warn(f"Falling back to random-init model: {exc}")
        model = ChessModel().eval()
        return model, True


# ---------- Activation capture ----------

def capture_activations(model: ChessModel, board_t: torch.Tensor,
                        move_t: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run a forward pass and return {layer_name: tensor}."""
    captures: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captures[name] = t.detach().float().cpu()
        return hook

    # Initial conv block - hook the ReLU so we get post-activation
    hooks.append(model.cnn[2].register_forward_hook(make_hook('initial_conv')))

    # ResidualBlock outputs
    block_idx = 0
    for module in model.cnn.children():
        if isinstance(module, ResidualBlock):
            hooks.append(module.register_forward_hook(
                make_hook(f'res_{block_idx}'))
            )
            block_idx += 1

    # LSTM (output of full sequence)
    hooks.append(model.lstm.register_forward_hook(make_hook('lstm')))

    # FC stack - hook each ReLU
    fc_relu_idx = 0
    for sub in model.fc.children():
        if isinstance(sub, nn.ReLU):
            hooks.append(sub.register_forward_hook(
                make_hook(f'fc_{fc_relu_idx}'))
            )
            fc_relu_idx += 1

    # Policy + value heads
    hooks.append(model.policy_head.register_forward_hook(make_hook('policy')))
    hooks.append(model.value_head.register_forward_hook(make_hook('value')))

    with torch.no_grad():
        model(board_t, move_t)

    for h in hooks:
        h.remove()

    return captures


def capture_lstm_per_step(model: ChessModel,
                          move_t: torch.Tensor) -> np.ndarray:
    """Step the LSTM one timestep at a time, return (T, hidden)."""
    model.lstm.eval()
    h = None
    states = []
    with torch.no_grad():
        for t in range(move_t.size(1)):
            x_t = move_t[:, t:t+1, :]
            out, h = model.lstm(x_t, h)
            states.append(h[0].squeeze(0).squeeze(0).cpu().numpy())
    return np.stack(states)  # (T, hidden)


# ---------- Building layer specs ----------

def _topk_abs(vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (selected_indices, normalized_activations) of length min(k, len(vec))."""
    k = min(k, vec.size)
    idx = np.argpartition(-np.abs(vec), k - 1)[:k]
    idx = idx[np.argsort(-np.abs(vec[idx]))]
    selected = vec[idx]
    return idx, selected


def _normalize(vec: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(vec))
    if m < 1e-9:
        return np.zeros_like(vec, dtype=np.float32)
    return (np.abs(vec) / m).astype(np.float32)


def build_layer_specs(captures: dict[str, torch.Tensor],
                      board_tensor: np.ndarray) -> list[LayerSpec]:
    """Convert raw activations into one LayerSpec per visualization column."""
    layers: list[LayerSpec] = []

    # Input: per-channel mean magnitude across spatial dims
    input_acts = np.abs(board_tensor).mean(axis=(0, 1))   # (17,)
    layers.append(LayerSpec('input', 'input', _normalize(input_acts),
                            color='#8aa9ff'))

    # Initial conv: per-channel L2 magnitude, subsample top-k
    init_conv = captures['initial_conv'].numpy()[0]       # (C, 8, 8)
    init_l2   = np.sqrt((init_conv ** 2).mean(axis=(1, 2)))
    init_idx, init_sel = _topk_abs(init_l2, MAX_NODES_PER_COLUMN)
    layers.append(LayerSpec('initial_conv', 'conv',
                            _normalize(init_sel), color='#9cc4ff'))

    # Residual blocks: use SAME channel indices across blocks so synapses
    # form coherent streams between columns
    block_keys = sorted(k for k in captures if k.startswith('res_'))
    for i, key in enumerate(block_keys):
        block = captures[key].numpy()[0]                  # (C, 8, 8)
        block_means = block.mean(axis=(1, 2))             # (C,)
        block_sel = block_means[init_idx]                 # reuse channel selection
        layers.append(LayerSpec(
            key, f'res{i+1}', _normalize(block_sel), color='#a8d4ff',
        ))

    # LSTM hidden state (final): all 64 hidden units
    lstm_h = captures['lstm'].numpy()[0]                  # (seq, hidden)
    lstm_final = lstm_h[-1] if lstm_h.ndim == 2 else lstm_h
    _, lstm_sel = _topk_abs(lstm_final, MAX_NODES_PER_COLUMN)
    layers.append(LayerSpec('lstm_hidden', 'lstm',
                            _normalize(lstm_sel), color='#c9b8ff'))

    # FC layers
    for i, key in enumerate(sorted(k for k in captures if k.startswith('fc_'))):
        fc = captures[key].numpy()[0]
        _, sel = _topk_abs(fc, MAX_NODES_PER_COLUMN)
        layers.append(LayerSpec(key, f'fc{i+1}',
                                _normalize(sel), color='#e4b4ff'))

    # Policy: top-8 by softmax probability
    policy_logits = captures['policy'].numpy()[0]
    policy_logits = policy_logits - policy_logits.max()   # numerical stability
    policy_probs = np.exp(policy_logits)
    policy_probs /= policy_probs.sum() + 1e-12
    top_k = 8
    p_idx = np.argpartition(-policy_probs, top_k - 1)[:top_k]
    p_idx = p_idx[np.argsort(-policy_probs[p_idx])]
    policy_spec = LayerSpec(
        'policy', 'policy',
        _normalize(policy_probs[p_idx]), color='#ffd07a',
    )
    policy_spec.top_indices = p_idx                       # type: ignore[attr-defined]
    policy_spec.top_probs = policy_probs[p_idx]           # type: ignore[attr-defined]
    layers.append(policy_spec)

    # Value: single node
    value_scalar = float(captures['value'].numpy().reshape(-1)[0])
    value_spec = LayerSpec(
        'value', 'value',
        np.array([abs(value_scalar)], dtype=np.float32), color='#ff8a8a',
    )
    value_spec.scalar = value_scalar                      # type: ignore[attr-defined]
    layers.append(value_spec)

    return layers


def position_layers(layers: list[LayerSpec], rng: np.random.Generator,
                    x_left: float = 0.23, x_right: float = 0.94,
                    y_low: float = 0.24, y_high: float = 0.82) -> None:
    """Assign x and y_positions to each layer. Mutates in place."""
    n = len(layers)
    for i, layer in enumerate(layers):
        layer.x = x_left + (x_right - x_left) * (i / max(1, n - 1))
        m = layer.activations.size
        if m == 1:
            layer.y_positions = np.array([0.5 * (y_low + y_high)])
        else:
            ys = np.linspace(y_low, y_high, m)
            jitter = rng.uniform(-0.012, 0.012, size=m)
            layer.y_positions = ys + jitter

        # Base node sizes encode activation magnitude
        base = 14 + 64 * layer.activations
        layer.node_sizes = base


# ---------- Synapses ----------

@dataclass
class SynapseSet:
    """Lines connecting layer i to layer i+1."""
    x_src: np.ndarray
    y_src: np.ndarray
    x_dst: np.ndarray
    y_dst: np.ndarray
    activations: np.ndarray  # combined src/dst strength, used for alpha


def build_synapses(layers: list[LayerSpec],
                   rng: np.random.Generator) -> list[SynapseSet]:
    sets: list[SynapseSet] = []
    for src, dst in zip(layers[:-1], layers[1:]):
        n_src, n_dst = src.activations.size, dst.activations.size
        all_pairs = n_src * n_dst
        if all_pairs <= MAX_SYNAPSES_PER_GAP:
            si, di = np.meshgrid(np.arange(n_src), np.arange(n_dst), indexing='ij')
            si, di = si.ravel(), di.ravel()
        else:
            si = rng.integers(0, n_src, size=MAX_SYNAPSES_PER_GAP)
            di = rng.integers(0, n_dst, size=MAX_SYNAPSES_PER_GAP)
        sets.append(SynapseSet(
            x_src=np.full(si.size, src.x),
            y_src=src.y_positions[si],
            x_dst=np.full(di.size, dst.x),
            y_dst=dst.y_positions[di],
            activations=0.5 * (src.activations[si] + dst.activations[di]),
        ))
    return sets


def pick_sparks(layers: list[LayerSpec],
                synapses: list[SynapseSet]) -> list[dict]:
    """Choose the brightest synapse(s) in each gap as 'spark' paths."""
    sparks = []
    for gap_idx, syn in enumerate(synapses):
        order = np.argsort(-syn.activations)
        chosen = order[:SPARKS_PER_GAP]
        sparks.append({
            'x_src': syn.x_src[chosen], 'y_src': syn.y_src[chosen],
            'x_dst': syn.x_dst[chosen], 'y_dst': syn.y_dst[chosen],
            'gap': gap_idx,
            'x_mid': 0.5 * (syn.x_src[chosen] + syn.x_dst[chosen]),
        })
    return sparks


# ---------- Chess board inset ----------

PIECE_FILENAME = {
    (chess.PAWN,   chess.WHITE): 'P_w.png',
    (chess.KNIGHT, chess.WHITE): 'N_w.png',
    (chess.BISHOP, chess.WHITE): 'B_w.png',
    (chess.ROOK,   chess.WHITE): 'R_w.png',
    (chess.QUEEN,  chess.WHITE): 'Q_w.png',
    (chess.KING,   chess.WHITE): 'K_w.png',
    (chess.PAWN,   chess.BLACK): 'P_b.png',
    (chess.KNIGHT, chess.BLACK): 'N_b.png',
    (chess.BISHOP, chess.BLACK): 'B_b.png',
    (chess.ROOK,   chess.BLACK): 'R_b.png',
    (chess.QUEEN,  chess.BLACK): 'Q_b.png',
    (chess.KING,   chess.BLACK): 'K_b.png',
}


def render_board_image(board: chess.Board, size_px: int = 256) -> np.ndarray:
    """Composite the board.png + piece pngs into a single RGBA array."""
    pieces_dir = REPO_ROOT / 'pieces'
    board_img = Image.open(pieces_dir / 'board.png').convert('RGBA').resize(
        (size_px, size_px), Image.LANCZOS,
    )
    cell = size_px // 8
    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        row = 7 - rank
        col = file
        piece_img = Image.open(
            pieces_dir / PIECE_FILENAME[(piece.piece_type, piece.color)]
        ).convert('RGBA').resize((cell, cell), Image.LANCZOS)
        board_img.alpha_composite(piece_img, (col * cell, row * cell))
    return np.array(board_img)


# ---------- Drawing ----------

def setup_figure(width: int, height: int):
    dpi = 100
    fig = plt.figure(
        figsize=(width / dpi, height / dpi),
        dpi=dpi, facecolor=BG_COLOR,
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax


def _gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))


def draw_static_chrome(ax, layers: list[LayerSpec], board: chess.Board,
                       used_random_init: bool, layout: LayoutConfig,
                       width: int, height: int):
    # Title typography reflects the active palette so the chrome matches
    # whichever brand frame the viz will live inside.
    if layout.show_title:
        title_text = 'Chess  CNN  +  LSTM'
        subtitle_text = (
            f'{RESIDUAL_BLOCKS} residual blocks   /   '
            'LSTM 132 -> 64   /   policy + value'
        )
        if TITLE_CAPS:
            title_text = title_text.upper()
            subtitle_text = subtitle_text.upper()
        ax.text(
            0.5, 0.945, title_text,
            ha='center', va='center', color=TITLE_COLOR,
            fontsize=22, fontweight=TITLE_WEIGHT, family=TITLE_FAMILY,
            fontstyle=TITLE_STYLE,
        )
        ax.text(
            0.5, 0.895, subtitle_text,
            ha='center', va='center', color=SUBTLE_COLOR,
            fontsize=8,
            family='monospace' if TITLE_FAMILY == 'monospace' else 'sans-serif',
        )

    if used_random_init and layout.show_title:
        ax.text(
            0.5, 0.005,
            'demo render — checkpoint architecture mismatch',
            ha='center', va='bottom', color=DEMO_COLOR,
            fontsize=7, family='sans-serif', alpha=0.85,
        )

    if layout.show_board:
        # Board inset on the left, kept square in rendered pixels.
        board_img = render_board_image(board, size_px=256)
        x0_b, x1_b = layout.board_x0, layout.board_x1
        y_height = (x1_b - x0_b) * (width / height)
        y_top = layout.board_y_top
        ax.imshow(
            board_img, extent=(x0_b, x1_b, y_top - y_height, y_top),
            zorder=5, interpolation='lanczos',
        )
        # Thin accent-color border matching the active palette
        ax.add_patch(mpatches.Rectangle(
            (x0_b, y_top - y_height), x1_b - x0_b, y_height,
            linewidth=0.6, edgecolor=BOARD_BORDER, facecolor='none',
            alpha=0.45, zorder=6,
        ))


def draw_readout(ax, policy_spec: LayerSpec, value_scalar: float,
                 board: chess.Board, flip: bool):
    """One elegant line below the network: best move + value.

    Replaces the dashboard-style panels with a single typeset readout
    that matches the restrained chess-UI aesthetic.
    """
    # Top-1 move from policy
    try:
        best_mv = policy_index_to_move(
            int(policy_spec.top_indices[0]),                # type: ignore[attr-defined]
            flip=flip,
        )
        best_label = (board.san(best_mv) if best_mv in board.legal_moves
                      else best_mv.uci())
    except Exception:
        best_label = '?'

    # Hidden text artists; reveal animations driven by update_frame.
    body_family = 'monospace' if TITLE_FAMILY == 'monospace' else 'sans-serif'
    move_caption_text = 'BEST MOVE' if TITLE_CAPS else 'best move'
    value_caption_text = 'VALUE (TANH)' if TITLE_CAPS else 'value (tanh)'
    move_label = ax.text(
        0.21, 0.135, '',
        ha='left', va='center', color=TITLE_COLOR,
        fontsize=15, family=TITLE_FAMILY,
        fontstyle=TITLE_STYLE, fontweight=TITLE_WEIGHT,
        zorder=6, alpha=0.0,
    )
    move_caption = ax.text(
        0.21, 0.085, move_caption_text,
        ha='left', va='center', color=FAINT_COLOR,
        fontsize=7.5, family=body_family,
        zorder=6, alpha=0.0,
    )
    value_label = ax.text(
        0.94, 0.135, '',
        ha='right', va='center', color=TITLE_COLOR,
        fontsize=15, family=TITLE_FAMILY,
        fontstyle=TITLE_STYLE, fontweight=TITLE_WEIGHT,
        zorder=6, alpha=0.0,
    )
    value_caption = ax.text(
        0.94, 0.085, value_caption_text,
        ha='right', va='center', color=FAINT_COLOR,
        fontsize=7.5, family=body_family,
        zorder=6, alpha=0.0,
    )
    return {
        'move_label': move_label, 'move_caption': move_caption,
        'value_label': value_label, 'value_caption': value_caption,
        'best_move_text': best_label,
        'value_scalar': float(value_scalar),
    }


def init_node_artists(ax, layers: list[LayerSpec]):
    """Three scatter passes per layer for a glow halo."""
    artists = []
    for layer in layers:
        base = layer.node_sizes
        color = NODE_CMAP(0.6) if layer.name != 'value' else (1, 1, 1, 1)
        # Halo (large, low alpha)
        halo = ax.scatter(
            [layer.x] * layer.activations.size, layer.y_positions,
            s=base * 12.0, c=[color], alpha=0.0,
            edgecolors='none', zorder=3,
        )
        # Mid glow
        mid = ax.scatter(
            [layer.x] * layer.activations.size, layer.y_positions,
            s=base * 4.0, c=[color], alpha=0.0,
            edgecolors='none', zorder=4,
        )
        # Core
        core = ax.scatter(
            [layer.x] * layer.activations.size, layer.y_positions,
            s=base, c=[color], alpha=0.95,
            edgecolors='none', zorder=5,
        )
        artists.append({'halo': halo, 'mid': mid, 'core': core,
                        'layer': layer})
    return artists


def init_synapse_artists(ax, synapses: list[SynapseSet]):
    line_collections = []
    for syn in synapses:
        segs = np.stack([
            np.stack([syn.x_src, syn.y_src], axis=1),
            np.stack([syn.x_dst, syn.y_dst], axis=1),
        ], axis=1)  # (N, 2, 2)
        widths = 0.4 + 1.6 * syn.activations
        lc = LineCollection(
            segs, colors=[(*EDGE_COLOR, 0.05)] * len(segs),
            linewidths=widths, zorder=2,
        )
        ax.add_collection(lc)
        line_collections.append(lc)
    return line_collections


def init_spark_artists(ax, sparks: list[dict]):
    artists = []
    for s in sparks:
        scatter = ax.scatter(
            s['x_src'], s['y_src'],
            s=24, c=[SPARK_COLOR], alpha=0.0,
            edgecolors='none', zorder=6,
        )
        artists.append(scatter)
    return artists


# ---------- Animation update ----------

def update_frame(frame: int, total_frames: int, layers, node_artists,
                 synapses, synapse_lcs, sparks, spark_artists, readout,
                 policy_x: float, value_x: float):
    t = frame / total_frames
    # Pulse sweeps from -0.05 to 1.05 over the middle 80% of the timeline
    sweep_start, sweep_end = 0.05, 0.85
    if t < sweep_start:
        p = -0.1
    elif t > sweep_end:
        p = 1.1
    else:
        p = -0.05 + 1.1 * (t - sweep_start) / (sweep_end - sweep_start)

    # Nodes: brightness = activation * (idle_floor + (1 - idle_floor) * envelope)
    for entry in node_artists:
        layer = entry['layer']
        env = _gaussian(layer.x, p, PULSE_SIGMA)
        brightness = layer.activations * (IDLE_FLOOR + (1 - IDLE_FLOOR) * env)
        colors = NODE_CMAP(0.30 + 0.60 * brightness)
        entry['core'].set_facecolor(colors)
        entry['mid'].set_facecolor(colors)
        entry['halo'].set_facecolor(colors)
        entry['core'].set_alpha(float(0.55 + 0.45 * env))
        entry['mid'].set_alpha(float(0.18 + 0.55 * env))
        entry['halo'].set_alpha(float(0.05 + 0.40 * env))

    # Synapses
    for syn, lc in zip(synapses, synapse_lcs):
        x_mid = 0.5 * (syn.x_src + syn.x_dst)
        env = _gaussian(x_mid, p, PULSE_SIGMA * 1.4)
        alphas = 0.05 + 0.60 * env * (0.3 + 0.7 * syn.activations)
        colors = np.tile(EDGE_COLOR, (alphas.size, 1))
        rgba = np.concatenate([colors, alphas[:, None]], axis=1)
        lc.set_color(rgba)

    # Sparks: position interpolated along source-destination line
    for s, art in zip(sparks, spark_artists):
        gap_xs = s['x_src'][0]
        gap_xe = s['x_dst'][0]
        if gap_xe == gap_xs:
            local_t = 0.5
        else:
            local_t = np.clip((p - gap_xs) / (gap_xe - gap_xs), 0.0, 1.0)
        env = _gaussian(s['x_mid'].mean(), p, PULSE_SIGMA * 1.2)
        xs = s['x_src'] + local_t * (s['x_dst'] - s['x_src'])
        ys = s['y_src'] + local_t * (s['y_dst'] - s['y_src'])
        art.set_offsets(np.column_stack([xs, ys]))
        art.set_alpha(float(0.95 * env))

    if readout is not None:
        # Readout: top move + value fade in as the pulse reaches the heads.
        move_reveal = max(0.0, min(1.0, (p - policy_x + 0.04) / 0.10))
        value_reveal = max(0.0, min(1.0, (p - value_x + 0.04) / 0.10))
        readout['move_label'].set_text(
            readout['best_move_text'] if move_reveal > 0 else ''
        )
        readout['move_label'].set_alpha(float(move_reveal))
        readout['move_caption'].set_alpha(float(move_reveal * 0.85))
        scalar = readout['value_scalar']
        settled = scalar * value_reveal
        readout['value_label'].set_text(f'{settled:+.2f}')
        readout['value_label'].set_alpha(float(value_reveal))
        readout['value_caption'].set_alpha(float(value_reveal * 0.85))


# ---------- Saving ----------

def render_all_frames(ax, fig, total_frames, update_fn) -> list[np.ndarray]:
    """Render every frame to an in-memory numpy RGB array."""
    frames = []
    for f in range(total_frames):
        update_fn(f)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(buf[..., :3].copy())
    return frames


def save_mp4(frames: list[np.ndarray], path: Path, fps: int):
    import imageio.v2 as imageio
    writer = imageio.get_writer(
        str(path), fps=fps, codec='libx264',
        quality=7, macro_block_size=1, pixelformat='yuv420p',
        ffmpeg_params=['-movflags', '+faststart'],
    )
    for fr in frames:
        writer.append_data(fr)
    writer.close()


def save_webm(frames: list[np.ndarray], path: Path, fps: int):
    import imageio.v2 as imageio
    writer = imageio.get_writer(
        str(path), fps=fps, codec='libvpx-vp9',
        ffmpeg_params=['-b:v', '0', '-crf', '36'],
        macro_block_size=1, pixelformat='yuv420p',
    )
    for fr in frames:
        writer.append_data(fr)
    writer.close()


def save_gif(frames: list[np.ndarray], path: Path, source_fps: int,
             max_w: int = 540, colors: int = 32, frame_stride: int = 2):
    """Downscale + quantize + frame-subsample, then write GIF.

    Three compression tricks stacked:
    1. Subsample frames (stride=2 by default) so the GIF holds half the
       motion data; total loop duration stays the same because we
       proportionally lower the GIF's playback fps.
    2. Quantize to a shared palette built from the richest frame so per-
       frame palettes don't bloat the file.
    3. Floyd-Steinberg dither to keep gradients smooth at low color counts.
    """
    if not frames:
        return
    sampled = frames[::frame_stride]
    effective_fps = max(1, source_fps // frame_stride)
    richest_idx = min(len(sampled) - 1, int(0.83 * len(sampled)))
    palette_src = Image.fromarray(sampled[richest_idx]).quantize(
        colors=colors, method=Image.Quantize.MEDIANCUT,
    )
    pil_frames = []
    for fr in sampled:
        img = Image.fromarray(fr)
        if img.width > max_w:
            new_h = int(img.height * max_w / img.width)
            img = img.resize((max_w, new_h), Image.LANCZOS)
        img = img.quantize(
            colors=colors, palette=palette_src,
            dither=Image.Dither.FLOYDSTEINBERG,
        )
        pil_frames.append(img)
    duration = int(1000 / effective_fps)
    pil_frames[0].save(
        path, save_all=True, append_images=pil_frames[1:],
        duration=duration, loop=0, optimize=True, disposal=2,
    )


def save_poster(frame: np.ndarray, path: Path):
    Image.fromarray(frame).save(path, optimize=True)


# ---------- Main ----------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fen', default=chess.STARTING_FEN,
                   help='Position to visualize (default: starting position).')
    p.add_argument('--moves', default='',
                   help='Optional UCI move list applied to --fen before viz '
                        '(comma-separated, e.g. "e2e4,e7e5,g1f3").')
    p.add_argument('--frames', type=int, default=180,
                   help='Total animation frames (default: 180 = 6s at 30fps).')
    p.add_argument('--fps', type=int, default=30, help='MP4/WebM fps.')
    p.add_argument('--width',  type=int, default=960)
    p.add_argument('--height', type=int, default=540)
    p.add_argument('--output-dir', default='media')
    p.add_argument('--layout', choices=sorted(LAYOUTS.keys()),
                   default='full',
                   help='Composition preset. "portfolio" removes the '
                        'internal title, readout, and chess board, then '
                        'positions the network around project-card overlays.')
    p.add_argument('--palette', choices=sorted(PALETTES.keys()),
                   default='terminal',
                   help='Color scheme: "terminal" matches the vmd306.com '
                        'portfolio (black + neon green + mono); "gold" '
                        'matches the chess web app (navy + gold + serif); '
                        '"redblue" is a diverging cool-to-hot palette.')
    p.add_argument('--suffix', default='',
                   help='Optional filename suffix, e.g. "_redblue" to keep '
                        'multiple palettes side by side in media/.')
    p.add_argument('--output-name', default='network',
                   help='Base filename stem for MP4/WebM/GIF outputs.')
    p.add_argument('--poster-name', default=None,
                   help='Poster filename stem. Defaults to '
                        '"<output-name>_poster".')
    p.add_argument('--no-mp4',  action='store_true')
    p.add_argument('--no-webm', action='store_true')
    p.add_argument('--no-gif',  action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(RNG_SEED)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Apply the chosen palette to module globals so all drawing helpers pick it up.
    global BG_COLOR, TITLE_COLOR, SUBTLE_COLOR, FAINT_COLOR
    global NODE_CMAP, EDGE_COLOR, SPARK_COLOR, BOARD_BORDER, DEMO_COLOR
    global TITLE_FAMILY, TITLE_STYLE, TITLE_WEIGHT, TITLE_CAPS
    pal = _resolve_palette(args.palette)
    BG_COLOR, TITLE_COLOR, SUBTLE_COLOR, FAINT_COLOR = (
        pal['bg'], pal['title'], pal['subtle'], pal['faint'])
    NODE_CMAP, EDGE_COLOR, SPARK_COLOR = (
        pal['cmap'], pal['edge_rgb'], pal['spark_rgb'])
    BOARD_BORDER, DEMO_COLOR = pal['board_border'], pal['demo_color']
    TITLE_FAMILY, TITLE_STYLE, TITLE_WEIGHT, TITLE_CAPS = (
        pal['title_family'], pal['title_style'],
        pal['title_weight'], pal['title_caps'])
    print(f'[info] Palette: {args.palette}')
    layout = LAYOUTS[args.layout]
    print(f'[info] Layout: {layout.name}')

    board = chess.Board(args.fen)
    if args.moves:
        for uci in args.moves.split(','):
            uci = uci.strip()
            if not uci:
                continue
            board.push(chess.Move.from_uci(uci))

    print(f'[info] Position: {board.fen()}')
    print(f'[info] Architecture: {RESIDUAL_BLOCKS}x ResNet ({RESIDUAL_FILTERS} filters)')

    model, used_random_init = load_model_or_random()
    board_t, move_t = _position_tensors(model, board)
    flip = board.turn == chess.BLACK

    print('[info] Capturing activations...')
    captures = capture_activations(model, board_t, move_t)
    board_tensor = board_t[0].cpu().numpy().transpose(1, 2, 0)
    layers = build_layer_specs(captures, board_tensor)
    position_layers(
        layers, rng,
        x_left=layout.layer_x_left,
        x_right=layout.layer_x_right,
        y_low=layout.layer_y_low,
        y_high=layout.layer_y_high,
    )

    synapses = build_synapses(layers, rng)
    sparks   = pick_sparks(layers, synapses)

    print(f'[info] Layers: {len(layers)} | synapse gaps: {len(synapses)}')

    fig, ax = setup_figure(args.width, args.height)
    draw_static_chrome(
        ax, layers, board, used_random_init, layout, args.width, args.height,
    )

    policy_spec = next(L for L in layers if L.name == 'policy')
    value_spec  = next(L for L in layers if L.name == 'value')

    readout = None
    if layout.show_readout:
        readout = draw_readout(
            ax, policy_spec,
            value_spec.scalar,                              # type: ignore[attr-defined]
            board, flip,
        )

    node_artists    = init_node_artists(ax, layers)
    synapse_lcs     = init_synapse_artists(ax, synapses)
    spark_artists   = init_spark_artists(ax, sparks)

    def update(frame):
        update_frame(frame, args.frames, layers, node_artists,
                     synapses, synapse_lcs, sparks, spark_artists,
                     readout, policy_spec.x, value_spec.x)

    print(f'[info] Rendering {args.frames} frames at {args.width}x{args.height}...')
    frames = render_all_frames(ax, fig, args.frames, update)

    sfx = args.suffix
    output_stem = f'{args.output_name}{sfx}'
    poster_base = args.poster_name or f'{args.output_name}_poster'
    poster_stem = f'{poster_base}{sfx}'
    poster_idx = min(len(frames) - 1, int(0.83 * args.frames))
    poster_path = out_dir / f'{poster_stem}.png'
    save_poster(frames[poster_idx], poster_path)
    _report(poster_path)

    if not args.no_mp4:
        mp4_path = out_dir / f'{output_stem}.mp4'
        save_mp4(frames, mp4_path, fps=args.fps)
        _report(mp4_path)
    if not args.no_webm:
        webm_path = out_dir / f'{output_stem}.webm'
        try:
            save_webm(frames, webm_path, fps=args.fps)
            _report(webm_path)
        except Exception as exc:
            print(f'[warn] WebM encoding failed: {exc}')
    if not args.no_gif:
        gif_path = out_dir / f'{output_stem}.gif'
        save_gif(frames, gif_path, source_fps=args.fps)
        _report(gif_path)

    print('[done]')


def _report(path: Path):
    size = path.stat().st_size
    if size < 1024:
        s = f'{size} B'
    elif size < 1024 ** 2:
        s = f'{size / 1024:.1f} KB'
    else:
        s = f'{size / (1024 ** 2):.2f} MB'
    print(f'  {path.name:24s}  {s}')


if __name__ == '__main__':
    main()
