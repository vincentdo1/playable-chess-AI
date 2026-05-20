export class PlayerControls {
  constructor(callbacks) {
    this.callbacks = callbacks;
    this.whiteSelect = document.getElementById('white-sel');
    this.blackSelect = document.getElementById('black-sel');
    this.depthGroup = document.getElementById('depth-grp');
    this.depthSlider = document.getElementById('depth-sl');
    this.depthValue = document.getElementById('depth-val');
    this.skillGroup = document.getElementById('skill-grp');
    this.skillSlider = document.getElementById('skill-sl');
    this.skillValue = document.getElementById('skill-val');
  }

  bind() {
    document.getElementById('btn-start').addEventListener('click', this.callbacks.onStart);
    document.getElementById('btn-reset').addEventListener('click', this.callbacks.onReset);
    document.getElementById('btn-go-new').addEventListener('click', this.callbacks.onReset);
    document.getElementById('btn-flip').addEventListener('click', this.callbacks.onFlip);
    document.getElementById('btn-undo').addEventListener('click', this.callbacks.onUndo);

    this.whiteSelect.addEventListener('change', () => this.syncEngineOptions());
    this.blackSelect.addEventListener('change', () => this.syncEngineOptions());

    this.depthSlider.addEventListener('input', () => {
      this.depthValue.textContent = this.depthSlider.value;
    });

    this.skillSlider.addEventListener('input', () => {
      this.skillValue.textContent = this.skillSlider.value;
    });

    this.skillSlider.addEventListener('change', () => {
      this.callbacks.onStockfishSkillChange(this.getStockfishSkill());
    });

    this.syncEngineOptions();
  }

  get whitePlayer() {
    return this.whiteSelect.value;
  }

  get blackPlayer() {
    return this.blackSelect.value;
  }

  currentPlayer(turn) {
    return turn === 'w' ? this.whitePlayer : this.blackPlayer;
  }

  getAlphabetaDepth() {
    return parseInt(this.depthSlider.value, 10);
  }

  getStockfishSkill() {
    return parseInt(this.skillSlider.value, 10);
  }

  enableMagnus() {
    ['w', 'b'].forEach((color) => {
      const option = document.getElementById(`mo-${color}`);
      if (!option) return;
      option.disabled = false;
      option.textContent = 'Magnus Carlsen';
    });
  }

  syncEngineOptions() {
    const usesAlphabeta = this.whitePlayer === 'alphabeta' || this.blackPlayer === 'alphabeta';
    const usesStockfish = this.whitePlayer === 'stockfish' || this.blackPlayer === 'stockfish';
    this.depthGroup.hidden = !usesAlphabeta;
    this.skillGroup.hidden = !usesStockfish;
  }
}
