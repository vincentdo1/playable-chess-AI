export class StockfishClient {
  constructor(stockfishPath, onBestMove) {
    this.stockfishPath = stockfishPath;
    this.onBestMove = onBestMove;
    this.worker = null;
  }

  ensureWorker(skillLevel) {
    if (this.worker) return true;

    try {
      this.worker = new Worker(this.stockfishPath);
    } catch (error) {
      console.error('Could not start Stockfish:', error);
      return false;
    }

    this.worker.onmessage = (event) => {
      if (typeof event.data !== 'string') return;
      if (!event.data.startsWith('bestmove')) return;

      const uci = event.data.split(' ')[1];
      this.onBestMove(uci && uci !== '(none)' ? uci : null);
    };

    this.worker.postMessage('uci');
    this.setSkill(skillLevel);
    this.worker.postMessage('isready');
    return true;
  }

  setSkill(skillLevel) {
    if (!this.worker) return;
    this.worker.postMessage(`setoption name Skill Level value ${skillLevel}`);
  }

  requestMove(fen, skillLevel, moveTimeMs = 600) {
    if (!this.ensureWorker(skillLevel)) return false;
    this.setSkill(skillLevel);
    this.worker.postMessage(`position fen ${fen}`);
    this.worker.postMessage(`go movetime ${moveTimeMs}`);
    return true;
  }

  restart(skillLevel) {
    if (this.worker) this.worker.terminate();
    this.worker = null;
    return this.ensureWorker(skillLevel);
  }
}
