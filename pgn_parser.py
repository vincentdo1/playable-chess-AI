import chess
import chess.pgn
import sys
import chess.engine

# Evaluates a board and returns CP and best move
def stockfish_evaluation(board, engine, time_limit = 0.0001):
    result = engine.analyse(board, chess.engine.Limit(time=time_limit))
    return result['score'].white(), result['pv'][0]
    
# Reads PGN file and adds evaluation, best move, and if best move was played for every move
def pgn_parse(path_open, path_save):
    pgn = open(path_open)
    mygame=chess.pgn.read_game(pgn)
    num = 1
    engine = chess.engine.SimpleEngine.popen_uci("C:/Users/vince/Downloads/scid_windows_5.0.2/scid_windows_x64/engines/stockfish.exe")
    f = open(path_save, "w")

    # Begin looping through every game
    while True:  
        # Begin looping through every move
        while mygame.next():
            fen = mygame.board().fen()
            board = chess.Board(fen)
            result = stockfish_evaluation(board, engine)
            node = mygame.variations[0]
            mygame=mygame.next()

            # Get board if best move was played
            board.push(result[1])

            # Check if previous board with best move played matches with current board
            best_move_played = board == chess.Board(mygame.board().fen())
            board.pop()
            node.comment = "[%" + "eval: " + str(result[0]) + "] [" + "%" + "best_move: " + str(result[1]) + "] [" + "%" + "played_best_move: " + str(best_move_played) + "]" 
        
        print("Printing game " + str(num) + " with evaluation")
        print(mygame.game(), file=f, end="\n\n")
        num += 1

        # Get next game in the PGN file
        mygame=chess.pgn.read_game(pgn)
        if mygame is None:
            break
    f.close()
    engine.close()
    pgn.close()
    print("COMPLETE")
    return

# Counts the number of times the best move was played in the PGN file
def read_pgn():
    pgn = open("c:/Users/vince/Downloads/magnus_evalv2.pgn", "r")
    mygame=chess.pgn.read_game(pgn)
    false = 0
    true = 0
    num = 0

    # Begin looping through every game
    while True:  
        # Begin looping through every move
        while mygame.next():
            fen = mygame.board().fen()
            board = chess.Board(fen)
            node = mygame.variations[0]

            # Check if it is Magnus' turn
            if (board.turn and mygame.game().headers["White"] == "Carlsen, Magnus" or not board.turn and mygame.game().headers["Black"] == "Carlsen, Magnus"):
                true += node.comment.count("True") # increment true counter
                false += node.comment.count("False") # increment false counter
            mygame=mygame.next()
        print("Finishing game " + str(num))
        num += 1

        mygame=chess.pgn.read_game(pgn)
        if mygame is None:
            break
    print("False: " + str(false))
    print("True: " + str(true))
    pgn.close()
if __name__ == '__main__':
    pgn_parse("c:/Users/vince/Downloads/GM_games_2600_pt2.pgn", "c:/Users/vince/Downloads/GM_games_eval.pgn")