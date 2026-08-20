"""Watch two trained checkpoints play each other in a Pygame window.

Reuses the board/piece art under pieces/. Each side moves via the raw policy
head or MCTS; moves are computed on a background thread so the window stays
responsive. Both checkpoints must use the current architecture â€” for an
old-architecture checkpoint on another branch use cross_model_match.py.

Controls: SPACE pause/resume, R restart, ESC/Q quit.
"""

from __future__ import annotations

import argparse
import threading
import time

import pygame
from PIL import Image
import chess


def make_move_fn(method, sims, batch_size, temperature, value_weight, value_candidates):
    """Return f(model, board) -> chess.Move. Imports torch-side lazily."""
    if method == 'mcts':
        from inference.mcts_player import mcts_search_best_move

        def fn(model, board):
            return mcts_search_best_move(
                model, board, num_simulations=sims, batch_size=batch_size,
            )
        return fn

    from load_model import predict_next_move

    def fn(model, board):
        uci = predict_next_move(
            model, board, temperature=temperature,
            value_weight=value_weight, value_candidate_limit=value_candidates,
        )
        return chess.Move.from_uci(uci) if uci else None
    return fn


class MoveWorker:
    """Compute the current side's move off the UI thread."""

    def __init__(self):
        self._thread = None
        self.result = None
        self.done = False

    def start(self, fn, model, board):
        self.done = False
        self.result = None
        snapshot = board.copy()

        def run():
            try:
                self.result = fn(model, snapshot)
            except Exception as exc:
                print(f"move computation error: {exc}")
                self.result = None
            finally:
                self.done = True

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    @property
    def running(self):
        return self._thread is not None and not self.done


class Arena:
    def __init__(self, args):
        pygame.init()
        self.size = args.window_size
        self.sq = self.size // 8
        self.screen = pygame.display.set_mode((self.size, self.size))
        pygame.display.set_caption('Model arena')
        self.clock = pygame.time.Clock()
        self.delay = args.delay
        self.max_plies = args.max_plies

        # Board background (reuse pieces/board.png like main.py).
        image = Image.open('pieces/board.png').resize((self.size, self.size))
        self.board_img = pygame.image.fromstring(
            image.tobytes(), image.size, image.mode,
        )

        # Cache scaled piece sprites: key = symbol like 'P_w'.
        self._piece_cache: dict[str, pygame.Surface] = {}

        from load_model import load_trained_model
        print(f"Loading White: {args.white_model}")
        self.white_model = load_trained_model(args.white_model)
        print(f"Loading Black: {args.black_model}")
        self.black_model = load_trained_model(args.black_model)

        self.white_fn = make_move_fn(
            args.white_method, args.sims, args.batch_size,
            args.temperature, args.value_weight, args.value_candidates,
        )
        self.black_fn = make_move_fn(
            args.black_method, args.sims, args.batch_size,
            args.temperature, args.value_weight, args.value_candidates,
        )
        self.white_label = f"{_short(args.white_model)} [{args.white_method}]"
        self.black_label = f"{_short(args.black_model)} [{args.black_method}]"

        self.reset()

    def reset(self):
        self.board = chess.Board()
        self.worker = MoveWorker()
        self.resume_at = 0.0
        self.result_text = None
        self.paused = False

    def _piece_sprite(self, piece):
        key = f"{piece.symbol().upper()}_{'w' if piece.color == chess.WHITE else 'b'}"
        if key not in self._piece_cache:
            img = pygame.image.load(f"pieces/{key}.png")
            self._piece_cache[key] = pygame.transform.scale(img, (self.sq, self.sq))
        return self._piece_cache[key]

    def to_pygame(self, square):
        col = chess.square_file(square)
        row = 7 - chess.square_rank(square)
        return col * self.sq, row * self.sq

    def draw(self):
        self.screen.blit(self.board_img, (0, 0))
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                sprite = self._piece_sprite(piece)
                x, y = self.to_pygame(square)
                self.screen.blit(sprite, (x, y))
        pygame.display.flip()

    def caption(self):
        if self.result_text:
            return self.result_text
        if self.paused:
            return 'PAUSED (space to resume)'
        side = 'White' if self.board.turn == chess.WHITE else 'Black'
        who = self.white_label if self.board.turn == chess.WHITE else self.black_label
        suffix = ' — thinking...' if self.worker.running else ''
        return f"Move {self.board.fullmove_number}: {side} {who}{suffix}"

    def step(self):
        over = (
            self.board.is_game_over(claim_draw=True)
            or len(self.board.move_stack) >= self.max_plies
        )
        if over:
            if self.result_text is None:
                outcome = self.board.outcome(claim_draw=True)
                if outcome is None:
                    self.result_text = 'Draw (move limit)'
                elif outcome.winner is None:
                    self.result_text = f'Draw ({outcome.termination.name.lower()})'
                else:
                    w = self.white_label if outcome.winner == chess.WHITE else self.black_label
                    self.result_text = f'{outcome.result()} — {w} wins'
                print(self.result_text)
            return

        if self.paused:
            return

        now = time.monotonic()
        if not self.worker.running and not self.worker.done and now >= self.resume_at:
            fn = self.white_fn if self.board.turn == chess.WHITE else self.black_fn
            model = self.white_model if self.board.turn == chess.WHITE else self.black_model
            self.worker.start(fn, model, self.board)

        if self.worker.done:
            move = self.worker.result
            if move is not None and move in self.board.legal_moves:
                san = self.board.san(move)
                self.board.push(move)
                print(f"{self.board.fullmove_number:>3}. {san}")
            else:
                self.result_text = 'Illegal/no move — aborted'
                print(self.result_text)
            self.resume_at = time.monotonic() + self.delay
            self.worker = MoveWorker()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.reset()
            self.step()
            self.draw()
            pygame.display.set_caption(self.caption())
            self.clock.tick(30)
        pygame.quit()


def _short(path):
    return path.replace('\\', '/').split('/')[-1]


def main():
    p = argparse.ArgumentParser(description='Watch two checkpoints play in Pygame.')
    p.add_argument('--white_model', required=True)
    p.add_argument('--black_model', required=True)
    p.add_argument('--white_method', choices=('policy', 'mcts'), default='mcts')
    p.add_argument('--black_method', choices=('policy', 'mcts'), default='mcts')
    p.add_argument('--sims', type=int, default=200, help='MCTS simulations per move.')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--temperature', type=float, default=0.0, help='Policy sampling temp.')
    p.add_argument('--value_weight', type=float, default=2.0)
    p.add_argument('--value_candidates', type=int, default=0)
    p.add_argument('--delay', type=float, default=0.4, help='Pause after each move (s).')
    p.add_argument('--max_plies', type=int, default=300)
    p.add_argument('--window_size', type=int, default=600)
    args = p.parse_args()
    Arena(args).run()


if __name__ == '__main__':
    main()
