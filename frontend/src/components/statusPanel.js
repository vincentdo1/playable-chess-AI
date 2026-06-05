const WHITE_PAWN = '\u2659';
const BLACK_PAWN = '\u265f';

const playerNames = {
  human: 'Human',
  alphabeta: 'Alphabeta AI',
  stockfish: 'Stockfish',
  random: 'Random',
  magnus: 'Magnus Carlsen',
};

export class StatusPanel {
  constructor() {
    this.connectionBadge = document.getElementById('conn-badge');
    this.connectionText = document.getElementById('conn-text');
    this.turnGlyph = document.getElementById('turn-glyph');
    this.statusText = document.getElementById('status-txt');
    this.statusSubtext = document.getElementById('status-sub');
    this.gameOverOverlay = document.getElementById('go-overlay');
    this.gameOverResult = document.getElementById('go-result');
    this.gameOverDetail = document.getElementById('go-detail');
  }

  setConnection(isOnline) {
    this.connectionBadge.className = `conn-badge ${isOnline ? 'online' : 'offline'}`;
    this.connectionText.textContent = isOnline ? 'Backend online' : 'Backend offline';
  }

  setThinking(isThinking) {
    this.turnGlyph.classList.toggle('pulse', isThinking);
  }

  reset() {
    this.statusText.textContent = 'Pick players and press Start.';
    this.statusSubtext.textContent = '';
    this.turnGlyph.textContent = WHITE_PAWN;
    this.gameOverOverlay.classList.remove('active');
    this.setThinking(false);
  }

  update(chess, currentPlayer) {
    if (chess.in_checkmate()) {
      this.statusText.textContent = `${chess.turn() === 'w' ? 'Black' : 'White'} wins by checkmate`;
      this.statusSubtext.textContent = '';
      return;
    }

    if (chess.in_draw()) {
      this.statusText.textContent = 'Draw';
      this.statusSubtext.textContent = chess.in_stalemate()
        ? 'Stalemate'
        : chess.insufficient_material()
          ? 'Insufficient material'
          : '50-move rule or repetition';
      return;
    }

    const side = chess.turn() === 'w' ? 'White' : 'Black';
    const moveNumber = Math.ceil(chess.history().length / 2) + 1;
    this.turnGlyph.textContent = chess.turn() === 'w' ? WHITE_PAWN : BLACK_PAWN;
    this.statusText.textContent = chess.in_check() ? `${side} is in check!` : `${side} to move`;
    this.statusSubtext.textContent = `${playerNames[currentPlayer] || currentPlayer} \u00b7 Move ${moveNumber}`;
  }

  showGameOver(chess) {
    if (chess.in_checkmate()) {
      this.gameOverResult.textContent = 'Checkmate';
      this.gameOverDetail.textContent = `${chess.turn() === 'w' ? 'Black' : 'White'} wins`;
    } else {
      this.gameOverResult.textContent = 'Draw';
      this.gameOverDetail.textContent = chess.in_stalemate()
        ? 'Stalemate'
        : '50-move rule or repetition';
    }
    this.setGameOverVisible(true);
  }

  setGameOverVisible(visible) {
    this.gameOverOverlay.classList.toggle('active', visible);
  }
}
