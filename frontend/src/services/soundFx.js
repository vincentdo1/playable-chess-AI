// Short synthesized move/capture ticks via the Web Audio API — no asset files,
// works offline. The context is created lazily and resumed on the first move so
// it satisfies browser autoplay rules (a move always follows a user gesture).
export class SoundFx {
  constructor() {
    this.ctx = null;
  }

  _ctx() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) this.ctx = new AudioCtx();
    }
    if (this.ctx && this.ctx.state === 'suspended') this.ctx.resume();
    return this.ctx;
  }

  _tick({ freq, drop, dur, peak }) {
    const ctx = this._ctx();
    if (!ctx) return;
    const t = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(freq, t);
    osc.frequency.exponentialRampToValueAtTime(Math.max(1, freq - drop), t + dur);
    gain.gain.setValueAtTime(peak, t);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    osc.connect(gain).connect(ctx.destination);
    osc.start(t);
    osc.stop(t + dur);
  }

  move() {
    this._tick({ freq: 380, drop: 140, dur: 0.09, peak: 0.10 });
  }

  capture() {
    this._tick({ freq: 260, drop: 120, dur: 0.12, peak: 0.14 });
  }
}
