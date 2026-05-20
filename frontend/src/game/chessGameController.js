export class ChessGameController {
  constructor({ api, boardView, controls, history, statusPanel, stockfish, config }) {
    this.api = api;
    this.boardView = boardView;
    this.controls = controls;
    this.history = history;
    this.statusPanel = statusPanel;
    this.stockfish = stockfish;
    this.config = config;
    this.chess = new window.Chess();
    this.running = false;
    this.thinking = false;
  }

  start() {
    this.reset();
    this.running = true;
    if (this.usesStockfish()) {
      this.stockfish.restart(this.controls.getStockfishSkill());
    }
    this.statusPanel.update(this.chess, this.currentPlayer());
    setTimeout(() => this.triggerAi(), 400);
  }

  reset() {
    this.chess.reset();
    this.boardView.start();
    this.boardView.resetHighlights();
    this.statusPanel.reset();
    this.history.reset();
    this.setThinking(false);
    this.running = false;
  }

  undo() {
    if (!this.running || this.thinking) return;

    this.chess.undo();
    this.chess.undo();
    this.boardView.setPosition(this.chess.fen());
    this.boardView.resetHighlights();
    this.statusPanel.update(this.chess, this.currentPlayer());
    this.history.render(this.chess);
  }

  currentPlayer() {
    return this.controls.currentPlayer(this.chess.turn());
  }

  usesStockfish() {
    return this.controls.whitePlayer === 'stockfish' ||
      this.controls.blackPlayer === 'stockfish';
  }

  onDragStart(source, piece) {
    if (!this.running || this.chess.game_over() || this.thinking) return false;

    const turn = this.chess.turn();
    if (this.currentPlayer() !== 'human') return false;
    if (turn === 'w' && piece[0] === 'b') return false;
    if (turn === 'b' && piece[0] === 'w') return false;

    this.showLegalMoves(source);
    return true;
  }

  onDrop(source, target) {
    this.boardView.clearLegalMoves();

    const move = this.chess.move({ from: source, to: target, promotion: 'q' });
    if (!move) return 'snapback';

    this.boardView.markLastMove(move.from, move.to);
    this.afterMove();
    return undefined;
  }

  onHover(square, piece) {
    if (!this.running || this.thinking || !piece) return;
    if (this.currentPlayer() !== 'human') return;

    const turn = this.chess.turn();
    if (turn === 'w' && piece[0] === 'b') return;
    if (turn === 'b' && piece[0] === 'w') return;
    this.showLegalMoves(square);
  }

  showLegalMoves(square) {
    this.boardView.showLegalMoves(this.chess.moves({ square, verbose: true }));
  }

  commitUci(uci) {
    if (!uci) {
      this.setThinking(false);
      return;
    }

    const move = this.chess.move({
      from: uci.slice(0, 2),
      to: uci.slice(2, 4),
      promotion: uci[4] || 'q',
    });

    if (!move) {
      this.setThinking(false);
      return;
    }

    this.boardView.markLastMove(move.from, move.to);
    this.boardView.setPosition(this.chess.fen());
    this.afterMove();
  }

  afterMove() {
    this.boardView.markCheck(this.chess);
    this.statusPanel.update(this.chess, this.currentPlayer());
    this.history.render(this.chess);
    this.setThinking(false);

    if (this.chess.game_over()) {
      this.running = false;
      this.statusPanel.showGameOver(this.chess);
      return;
    }

    setTimeout(() => this.triggerAi(), 280);
  }

  async triggerAi() {
    if (!this.running || this.chess.game_over() || this.thinking) return;

    const player = this.currentPlayer();
    if (player === 'human') return;

    this.setThinking(true);

    if (player === 'random') {
      setTimeout(() => this.playRandomMove(), 220);
      return;
    }

    if (player === 'stockfish') {
      const requested = this.stockfish.requestMove(
        this.chess.fen(),
        this.controls.getStockfishSkill()
      );
      if (!requested) this.setThinking(false);
      return;
    }

    try {
      const response = await this.api.move(this.buildServerMoveRequest(player));
      if (response.move) this.commitUci(response.move);
      else {
        console.error('API error:', response.error);
        this.setThinking(false);
      }
    } catch (error) {
      console.error('API failed:', error);
      this.setThinking(false);
    }
  }

  playRandomMove() {
    const moves = this.chess.moves();
    if (!moves.length) {
      this.setThinking(false);
      return;
    }

    const move = this.chess.move(moves[Math.floor(Math.random() * moves.length)]);
    if (!move) {
      this.setThinking(false);
      return;
    }

    this.boardView.markLastMove(move.from, move.to);
    this.boardView.setPosition(this.chess.fen());
    this.afterMove();
  }

  buildServerMoveRequest(player) {
    const payload = { fen: this.chess.fen(), player };

    if (player === 'alphabeta') {
      payload.depth = this.controls.getAlphabetaDepth();
    }

    if (player === 'magnus') {
      payload.temperature = this.config.magnus.temperature;
      payload.value_weight = this.config.magnus.valueWeight;
      payload.value_candidates = this.config.magnus.valueCandidates;
    }

    return payload;
  }

  setThinking(isThinking) {
    this.thinking = isThinking;
    this.boardView.setThinking(isThinking);
    this.statusPanel.setThinking(isThinking);
  }
}
