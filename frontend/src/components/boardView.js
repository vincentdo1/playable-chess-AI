const squareElement = (square) => document.querySelector(`.square-${square}`);

export class BoardView {
  constructor({ piecesPath, callbacks }) {
    this.piecesPath = piecesPath;
    this.callbacks = callbacks;
    this.board = null;
    this.lastFrom = null;
    this.lastTo = null;
    this.selectedSquare = null;
    this.legalSquares = [];
  }

  init() {
    this.board = window.Chessboard('board', {
      draggable: true,
      position: 'start',
      pieceTheme: (piece) => {
        const color = piece[0] === 'w' ? '_w' : '_b';
        return `${this.piecesPath}/${piece[1]}${color}.png`;
      },
      onDragStart: (...args) => this.callbacks.onDragStart(...args),
      onDrop: (...args) => this.callbacks.onDrop(...args),
      onSnapEnd: () => this.callbacks.onSnapEnd(),
      onMouseoverSquare: (...args) => this.callbacks.onHover(...args),
      onMouseoutSquare: () => this.callbacks.onMouseout(),
    });

    // Click-to-move: chessboard.js has no per-square click hook, so delegate
    // from the board element and read the algebraic square it carries.
    document.getElementById('board').addEventListener('click', (event) => {
      const squareEl = event.target.closest('.square-55d63');
      if (!squareEl) return;
      const square = squareEl.getAttribute('data-square')
        || (squareEl.className.match(/square-([a-h][1-8])/) || [])[1];
      if (square) this.callbacks.onSquareClick(square);
    });
  }

  start() {
    this.board.start();
  }

  flip() {
    this.board.flip();
  }

  setPosition(fen) {
    this.board.position(fen);
  }

  setThinking(isThinking) {
    document.getElementById('think-overlay').classList.toggle('active', isThinking);
    document.getElementById('board-frame').classList.toggle('thinking', isThinking);
  }

  showLegalMoves(moves) {
    this.clearLegalMoves();
    moves.forEach((move) => {
      squareElement(move.to)?.classList.add('highlight-legal');
      this.legalSquares.push(move.to);
    });
  }

  clearLegalMoves() {
    this.legalSquares.forEach((square) => {
      squareElement(square)?.classList.remove('highlight-legal');
    });
    this.legalSquares = [];
  }

  markLastMove(from, to) {
    this.clearLastMove();
    this.lastFrom = from;
    this.lastTo = to;
    squareElement(from)?.classList.add('highlight-last');
    squareElement(to)?.classList.add('highlight-last');
  }

  clearLastMove() {
    squareElement(this.lastFrom)?.classList.remove('highlight-last');
    squareElement(this.lastTo)?.classList.remove('highlight-last');
    this.lastFrom = null;
    this.lastTo = null;
  }

  markSelected(square) {
    this.clearSelected();
    this.selectedSquare = square;
    squareElement(square)?.classList.add('highlight-selected');
  }

  clearSelected() {
    squareElement(this.selectedSquare)?.classList.remove('highlight-selected');
    this.selectedSquare = null;
  }

  markCheck(chess) {
    this.clearCheck();
    if (!chess.in_check()) return;

    const color = chess.turn();
    chess.board().forEach((row, rankIndex) => {
      row.forEach((piece, fileIndex) => {
        if (!piece || piece.type !== 'k' || piece.color !== color) return;
        const square = `${String.fromCharCode(97 + fileIndex)}${8 - rankIndex}`;
        squareElement(square)?.classList.add('highlight-check');
      });
    });
  }

  clearCheck() {
    document.querySelectorAll('.highlight-check').forEach((element) => {
      element.classList.remove('highlight-check');
    });
  }

  resetHighlights() {
    this.clearLastMove();
    this.clearLegalMoves();
    this.clearCheck();
    this.clearSelected();
  }
}
