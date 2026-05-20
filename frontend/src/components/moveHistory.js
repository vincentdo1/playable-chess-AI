export class MoveHistory {
  constructor() {
    this.list = document.getElementById('hist-list');
  }

  reset() {
    this.list.innerHTML = '<div class="history-empty">No moves yet</div>';
  }

  render(chess) {
    const history = chess.history();
    if (!history.length) {
      this.reset();
      return;
    }

    this.list.innerHTML = history
      .reduce((rows, move, index) => {
        if (index % 2 === 0) {
          rows.push({
            number: Math.floor(index / 2) + 1,
            white: move,
            black: '',
          });
        } else {
          rows[rows.length - 1].black = move;
        }
        return rows;
      }, [])
      .map((row) => `
        <div class="move-pair">
          <span class="mn">${row.number}.</span>
          <span class="mw">${row.white}</span>
          <span class="mb">${row.black}</span>
        </div>
      `)
      .join('');

    const scroll = this.list.parentElement;
    scroll.scrollTop = scroll.scrollHeight;
  }
}
