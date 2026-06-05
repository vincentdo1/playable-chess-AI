export class ChessGameController {
  constructor({ api, boardView, controls, history, statusPanel, stockfish, sound, config }) {
    this.api = api;
    this.boardView = boardView;
    this.controls = controls;
    this.history = history;
    this.statusPanel = statusPanel;
    this.stockfish = stockfish;
    this.sound = sound;
    this.config = config;
    this.chess = new window.Chess();
    this.running = false;
    this.thinking = false;
    this.selectedSquare = null;
    this.gameOver = false;
    this.reviewPly = null;
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
    this.clearSelection();
    this.clearReview();
    this.statusPanel.reset();
    this.history.reset();
    this.controls.setResultButtonVisible(false);
    this.gameOver = false;
    this.setThinking(false);
    this.running = false;
  }

  undo() {
    if (!this.running || this.thinking) return;

    this.chess.undo();
    this.chess.undo();
    this.clearSelection();
    this.clearReview();
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
    if (this.reviewPly !== null) return false;

    const turn = this.chess.turn();
    if (this.currentPlayer() !== 'human') return false;
    if (turn === 'w' && piece[0] === 'b') return false;
    if (turn === 'b' && piece[0] === 'w') return false;

    this.clearSelection();
    this.showLegalMoves(source);
    return true;
  }

  onDrop(source, target) {
    this.boardView.clearLegalMoves();

    const move = this.chess.move({ from: source, to: target, promotion: 'q' });
    if (!move) return 'snapback';

    this.boardView.markLastMove(move.from, move.to);
    this.afterMove(move);
    return undefined;
  }

  onHover(square, piece) {
    if (!this.running || this.thinking || !piece) return;
    if (this.currentPlayer() !== 'human') return;
    if (this.selectedSquare) return;

    const turn = this.chess.turn();
    if (turn === 'w' && piece[0] === 'b') return;
    if (turn === 'b' && piece[0] === 'w') return;
    this.showLegalMoves(square);
  }

  onMouseout() {
    if (this.selectedSquare) return;
    this.boardView.clearLegalMoves();
  }

  onSquareClick(square) {
    if (!this.running || this.chess.game_over() || this.thinking) return;
    if (this.reviewPly !== null) return;
    if (this.currentPlayer() !== 'human') return;

    const piece = this.chess.get(square);
    const turn = this.chess.turn();

    if (this.selectedSquare) {
      if (square === this.selectedSquare) {
        this.clearSelection();
        return;
      }

      const move = this.chess.move({ from: this.selectedSquare, to: square, promotion: 'q' });
      if (move) {
        this.clearSelection();
        this.boardView.setPosition(this.chess.fen());
        this.boardView.markLastMove(move.from, move.to);
        this.afterMove(move);
        return;
      }

      // Illegal target: reselect another own piece, otherwise drop the selection.
      if (piece && piece.color === turn) this.selectSquare(square);
      else this.clearSelection();
      return;
    }

    if (piece && piece.color === turn) this.selectSquare(square);
  }

  selectSquare(square) {
    this.selectedSquare = square;
    this.boardView.markSelected(square);
    this.showLegalMoves(square);
  }

  clearSelection() {
    this.selectedSquare = null;
    this.boardView.clearSelected();
    this.boardView.clearLegalMoves();
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

    if (this.reviewPly === null) {
      this.boardView.markLastMove(move.from, move.to);
      this.boardView.setPosition(this.chess.fen());
    }
    this.afterMove(move);
  }

  afterMove(move) {
    this.playMoveSound(move);
    if (this.reviewPly === null) this.boardView.markCheck(this.chess);
    this.statusPanel.update(this.chess, this.currentPlayer());
    this.history.render(this.chess);
    this.setThinking(false);

    if (this.chess.game_over()) {
      this.running = false;
      this.gameOver = true;
      this.clearSelection();
      this.statusPanel.showGameOver(this.chess);
      this.controls.setResultButtonVisible(false);
      return;
    }

    setTimeout(() => this.triggerAi(), 280);
  }

  dismissGameOver() {
    if (!this.gameOver) return;
    this.statusPanel.setGameOverVisible(false);
    this.controls.setResultButtonVisible(true);
  }

  showResult() {
    if (!this.gameOver) return;
    this.statusPanel.setGameOverVisible(true);
    this.controls.setResultButtonVisible(false);
  }

  playMoveSound(move) {
    if (!move || !this.sound) return;
    const isCapture = !!move.captured || (move.flags && /[ce]/.test(move.flags));
    if (isCapture) this.sound.capture();
    else this.sound.move();
  }

  // History scrubbing. ply = number of half-moves shown (0 = start position,
  // history length = the live/latest position, which exits review).
  goToPly(ply) {
    const moves = this.chess.history({ verbose: true });
    if (!moves.length) return;
    const target = Math.max(0, Math.min(ply, moves.length));

    if (target >= moves.length) {
      this.clearReview();
      this.boardView.setPosition(this.chess.fen());
      this.boardView.resetHighlights();
      const last = moves[moves.length - 1];
      this.boardView.markLastMove(last.from, last.to);
      this.boardView.markCheck(this.chess);
      return;
    }

    this.reviewPly = target;
    this.clearSelection();

    const replay = new window.Chess();
    for (let i = 0; i < target; i++) replay.move(moves[i].san);
    this.boardView.setPosition(replay.fen());
    this.boardView.resetHighlights();
    if (target > 0) {
      const m = moves[target - 1];
      this.boardView.markLastMove(m.from, m.to);
    }
    this.boardView.markCheck(replay);
    this.history.setActivePly(target);
  }

  stepBack() {
    const total = this.chess.history().length;
    if (!total) return;
    const current = this.reviewPly === null ? total : this.reviewPly;
    this.goToPly(current - 1);
  }

  stepForward() {
    if (this.reviewPly === null) return;
    this.goToPly(this.reviewPly + 1);
  }

  clearReview() {
    this.reviewPly = null;
    this.history.setActivePly(null);
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

    if (this.reviewPly === null) {
      this.boardView.markLastMove(move.from, move.to);
      this.boardView.setPosition(this.chess.fen());
    }
    this.afterMove(move);
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
      if (this.controls.getMagnusUseMcts()) {
        payload.use_mcts = true;
        payload.mcts_simulations = this.config.magnus.mctsSimulations;
      }
    }

    return payload;
  }

  setThinking(isThinking) {
    this.thinking = isThinking;
    this.boardView.setThinking(isThinking);
    this.statusPanel.setThinking(isThinking);
  }
}
