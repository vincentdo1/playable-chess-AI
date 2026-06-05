import { ChessApi } from './api/chessApi.js';
import { BoardView } from './components/boardView.js';
import { MoveHistory } from './components/moveHistory.js';
import { PlayerControls } from './components/playerControls.js';
import { StatusPanel } from './components/statusPanel.js';
import { config } from './config.js';
import { ChessGameController } from './game/chessGameController.js';
import { StockfishClient } from './services/stockfishClient.js';
import { SoundFx } from './services/soundFx.js';

const api = new ChessApi(config.apiUrl);
const statusPanel = new StatusPanel();
const sound = new SoundFx();

let game;

const history = new MoveHistory((ply) => game.goToPly(ply));

const boardView = new BoardView({
  piecesPath: config.piecesPath,
  callbacks: {
    onDragStart: (...args) => game.onDragStart(...args),
    onDrop: (...args) => game.onDrop(...args),
    onSnapEnd: () => boardView.setPosition(game.chess.fen()),
    onHover: (...args) => game.onHover(...args),
    onMouseout: () => game.onMouseout(),
    onSquareClick: (square) => game.onSquareClick(square),
  },
});

const stockfish = new StockfishClient(
  config.stockfishPath,
  (uci) => game.commitUci(uci)
);

const controls = new PlayerControls({
  onStart: () => game.start(),
  onReset: () => game.reset(),
  onFlip: () => boardView.flip(),
  onUndo: () => game.undo(),
  onStockfishSkillChange: (skill) => stockfish.setSkill(skill),
  onDismissGameOver: () => game.dismissGameOver(),
  onShowResult: () => game.showResult(),
  onStepBack: () => game.stepBack(),
  onStepForward: () => game.stepForward(),
});

game = new ChessGameController({
  api,
  boardView,
  controls,
  history,
  statusPanel,
  stockfish,
  sound,
  config,
});

boardView.init();
controls.bind();
checkBackend();

async function checkBackend() {
  try {
    const data = await api.health();
    statusPanel.setConnection(true);
    if (data.players?.magnus) controls.enableMagnus();
  } catch (error) {
    statusPanel.setConnection(false);
  }
}
