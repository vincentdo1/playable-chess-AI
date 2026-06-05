export class MoveHistory {
  constructor(onSelectPly) {
    this.list = document.getElementById('hist-list');
    this.onSelectPly = onSelectPly;
    this.activePly = null;

    this.list.addEventListener('click', (event) => {
      const el = event.target.closest('[data-ply]');
      if (el) this.onSelectPly?.(Number(el.getAttribute('data-ply')));
    });
  }

  reset() {
    this.activePly = null;
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
            number: index / 2 + 1,
            white: move,
            whitePly: index + 1,
            black: '',
            blackPly: null,
          });
        } else {
          rows[rows.length - 1].black = move;
          rows[rows.length - 1].blackPly = index + 1;
        }
        return rows;
      }, [])
      .map((row) => `
        <div class="move-pair">
          <span class="mn">${row.number}.</span>
          <span class="mw" data-ply="${row.whitePly}">${row.white}</span>
          ${row.black
            ? `<span class="mb" data-ply="${row.blackPly}">${row.black}</span>`
            : '<span class="mb"></span>'}
        </div>
      `)
      .join('');

    this.applyActive();

    const scroll = this.list.parentElement;
    scroll.scrollTop = scroll.scrollHeight;
  }

  setActivePly(ply) {
    this.activePly = ply;
    this.applyActive();
  }

  applyActive() {
    this.list.querySelectorAll('[data-ply]').forEach((el) => {
      el.classList.toggle('active', Number(el.getAttribute('data-ply')) === this.activePly);
    });
  }
}
