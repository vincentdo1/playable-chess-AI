import os, sys, argparse
import pygame
from PIL import Image
import chess
import chess_player

# Vincent Do, April 2026
class ChessGame:
    def __init__(self, white_player="you", black_player="random"):
        pygame.init()
        self.WINDOW_SIZE = 600
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
        pygame.display.set_caption('Chessboard')
        self.clock = pygame.time.Clock()

        board_image_path = "pieces/board.png"
        image = Image.open(board_image_path)
        image = image.resize((self.WINDOW_SIZE, self.WINDOW_SIZE))
        mode = image.mode
        size = image.size
        data = image.tobytes()
        self.chessboard_img = pygame.image.fromstring(data, size, mode)

        self.board = chess.Board()
        self.selected_square = None
        self.player_color = chess.WHITE if white_player == "you" else chess.BLACK
        self.ai_only = True if white_player != "you" and black_player != "you" else False
        self.white_player = white_player
        self.black_player = black_player

        # Load Magnus model lazily — only when actually needed.
        self._magnus_model = None
        self._predict_fn   = None
        if white_player == "magnus_carlsen" or black_player == "magnus_carlsen":
            try:
                from load_model import load_trained_model, predict_next_move_with_search as predict_next_move
                self._magnus_model = load_trained_model()
                self._predict_fn   = predict_next_move
            except Exception as e:
                print(f"Warning: could not load Magnus model ({e}). "
                      f"'magnus_carlsen' will fall back to random.")

    def to_chess_coords(self, pygame_coords):
        x, y = pygame_coords
        col = x // (self.WINDOW_SIZE // 8)
        row = 7 - y // (self.WINDOW_SIZE // 8)
        return chess.square(col, row)

    def to_pygame_coords(self, chess_coords):
        col = chess.square_file(chess_coords)
        row = 7 - chess.square_rank(chess_coords)
        return col * (self.WINDOW_SIZE // 8), row * (self.WINDOW_SIZE // 8)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                square = self.to_chess_coords((x, y))
                piece = self.board.piece_at(square)
                if piece is not None:
                    self.selected_square = square
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if self.selected_square is not None:
                    x, y = event.pos
                    target_square = self.to_chess_coords((x, y))
                    move = chess.Move(self.selected_square, target_square)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                    self.selected_square = None
        return True

    def draw(self):
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.chessboard_img, (0, 0))
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                color_suffix = "_w" if piece.color == chess.WHITE else "_b"
                piece_image = pygame.image.load(f"pieces/{piece.symbol()}{color_suffix}.png")
                piece_image = pygame.transform.scale(piece_image,
                    (self.WINDOW_SIZE // 8, self.WINDOW_SIZE // 8))
                x, y = self.to_pygame_coords(square)
                piece_x = x + (self.WINDOW_SIZE // 8 - piece_image.get_width()) // 2
                piece_y = y + (self.WINDOW_SIZE // 8 - piece_image.get_height()) // 2
                self.screen.blit(piece_image, (piece_x, piece_y))
        pygame.display.flip()

    def update(self):
        if self.board.is_game_over():
            return

        # Dispatch based on whose turn it is — argument order never matters.
        current = self.white_player if self.board.turn == chess.WHITE else self.black_player

        if current == "you":
            return  # human move is handled by handle_events()

        elif current == "engine":
            ai_move = chess_player.get_best_move(self.board, 1)

        elif current == "random":
            ai_move = chess_player.random_move_player(self.board)

        elif current == "alphabeta":
            ai_move = chess_player.alphabeta(self.board.turn, self.board, 3)

        elif current == "magnus_carlsen":
            if self._magnus_model is not None:
                uci = self._predict_fn(self._magnus_model, self.board, top_n=10, depth=4)
                ai_move = chess.Move.from_uci(uci) if uci else chess_player.random_move_player(self.board)
            else:
                ai_move = chess_player.random_move_player(self.board)

        else:
            return

        self.board.push(ai_move)

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.draw()
            self.update()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description     = 'AI Chess Player',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--white_player', default='you',
                        choices=('random', 'you', 'engine', 'alphabeta', 'magnus_carlsen'))
    parser.add_argument('--black_player', default='you',
                        choices=('random', 'you', 'engine', 'alphabeta', 'magnus_carlsen'))
    args = parser.parse_args()
    application = ChessGame(args.white_player, args.black_player)
    application.run()


if __name__ == "__main__":
    main()