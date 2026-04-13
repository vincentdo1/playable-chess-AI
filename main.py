import os,sys,argparse
import pygame
from PIL import Image
import chess
import chess_player

# Import the neural network player — only loads the model once at startup
try:
    from load_model import load_trained_model, predict_next_move
    _magnus_model = load_trained_model()
except Exception as e:
    _magnus_model = None
    print(f"Warning: could not load Magnus model ({e}). 'magnus_carlsen' player will fall back to random.")

################################################################################################
# Main chess-playing application.
# Vincent Do, July 2023
#
class ChessGame:
    def __init__(self, white_player="you", black_player="random"):
        # Set up the Pygame window
        pygame.init()
        self.WINDOW_SIZE = 600
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
        pygame.display.set_caption('Chessboard')
        self.clock = pygame.time.Clock()

        # Load the chessboard image from your computer
        board_image_path = "pieces/board.png"  # Replace this with the file path to the downloaded chessboard image
        image = Image.open(board_image_path)
        image = image.resize((self.WINDOW_SIZE, self.WINDOW_SIZE))
            
        # Convert the PIL image to a Pygame surface
        mode = image.mode
        size = image.size
        data = image.tobytes()
        self.chessboard_img = pygame.image.fromstring(data, size, mode)

        # Initialize the chessboard and other variables
        self.board = chess.Board()
        self.selected_square = None

        # Determine player color
        self.player_color = chess.WHITE if white_player == "you" else chess.BLACK
        self.ai_only = True if white_player != "you" and black_player != "you" else False
        self.white_player = white_player
        self.black_player = black_player

    # Function to convert Pygame coordinates to chessboard coordinates
    def to_chess_coords(self, pygame_coords):
        x, y = pygame_coords
        col = x // (self.WINDOW_SIZE // 8)
        row = 7 - y // (self.WINDOW_SIZE // 8)
        return chess.square(col, row)

    # Function to convert chessboard coordinates to Pygame coordinates
    def to_pygame_coords(self, chess_coords):
        col = chess.square_file(chess_coords)
        row = 7 - chess.square_rank(chess_coords)
        return col * (self.WINDOW_SIZE // 8), row * (self.WINDOW_SIZE // 8)

    def handle_events(self):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left mouse button click
                    x, y = event.pos
                    square = self.to_chess_coords((x, y))
                    piece = self.board.piece_at(square)
                    if piece is not None:
                        self.selected_square = square
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:  # Left mouse button release
                    if self.selected_square is not None:
                        x, y = event.pos
                        target_square = self.to_chess_coords((x, y))
                        move = chess.Move(self.selected_square, target_square)
                        if move in self.board.legal_moves:
                            self.board.push(move)
                        self.selected_square = None
            return True
            
    # Function to draw the game on the screen
    def draw(self):
        # Clear the screen and draw the chessboard image
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.chessboard_img, (0, 0))

        # Draw the pieces on the board
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                # Determine the color of the piece
                color_suffix = "_w" if piece.color == chess.WHITE else "_b"

                # Load the piece image based on the filename with color suffix
                piece_image = pygame.image.load(f"pieces/{piece.symbol()}{color_suffix}.png")
                piece_image = pygame.transform.scale(piece_image, (self.WINDOW_SIZE // 8, self.WINDOW_SIZE // 8))

                # Calculate the position to center the piece image on the square
                x, y = self.to_pygame_coords(square)
                piece_x = x + (self.WINDOW_SIZE // 8 - piece_image.get_width()) // 2
                piece_y = y + (self.WINDOW_SIZE // 8 - piece_image.get_height()) // 2
                    
                self.screen.blit(piece_image, (piece_x, piece_y))

        # Update the display
        pygame.display.flip()

    # Function to update the game state
    def update(self):
        # Check if it's the AI's turn and get its move
        if not self.board.is_game_over():
            if self.board.turn and self.white_player == "engine" or not self.board.turn and self.black_player == "engine":
                ai_move = chess_player.get_best_move(self.board, 1)
                self.board.push(ai_move)
            elif self.board.turn and self.white_player == "random" or not self.board.turn and self.black_player == "random":
                ai_move = chess_player.random_move_player(self.board)
                self.board.push(ai_move)
            elif self.board.turn and self.white_player == "alphabeta" or not self.board.turn and self.black_player == "alphabeta":
                ai_move = chess_player.alphabeta(self.board.turn, self.board, 3)
                self.board.push(ai_move)
            elif self.board.turn and self.white_player == "magnus_carlsen" or not self.board.turn and self.black_player == "magnus_carlsen":
                if _magnus_model is not None:
                    uci = predict_next_move(_magnus_model, self.board)
                    ai_move = chess.Move.from_uci(uci) if uci else chess_player.random_move_player(self.board)
                else:
                    ai_move = chess_player.random_move_player(self.board)
                self.board.push(ai_move)

    # Function to run the game loop
    def run(self):
        running = True
        while running:
            running = self.handle_events()

            # Draw the game on the screen
            self.draw()

            # Update the game state
            self.update()

            # Update the display
            pygame.display.flip()

            # Limit the frame rate
            self.clock.tick(60)

        # Quit Pygame
        pygame.quit()

def main():
    parser = argparse.ArgumentParser(
        description     = 'AI Chess Player', 
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--white_player', default = 'you',
                        choices = ('random', 'you', 'engine', 'alphabeta', 'magnus_carlsen'))
    parser.add_argument('--black_player', default = 'you',
                        choices = ('random', 'you', 'engine', 'alphabeta', 'magnus_carlsen'))
    args   = parser.parse_args()
    application = ChessGame(args.white_player, args.black_player)
    application.run()


if __name__ == "__main__":
    main()