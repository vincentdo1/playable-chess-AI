import { ChessApi } from './api/chessApi.js';
import { BoardView } from './components/boardView.js';
import { MoveHistory } from './components/moveHistory.js';
import { PlayerControls } from './components/playerControls.js';
import { StatusPanel } from './components/statusPanel.js';
import { config } from './config.js';
import { ChessGameController } from './game/chessGameController.js';
import { StockfishClient } from './services/stockfishClient.js';

const api = new ChessApi(config.apiUrl);
const statusPanel = new StatusPanel();
const history = new MoveHistory();

let game;

const boardView = new BoardView({
  piecesPath: config.piecesPath,
  callbacks: {
    onDragStart: (...args) => game.onDragStart(...args),
    onDrop: (...args) => game.onDrop(...args),
    onSnapEnd: () => boardView.setPosition(game.chess.fen()),
    onHover: (...args) => game.onHover(...args),
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
});

game = new ChessGameController({
  api,
  boardView,
  controls,
  history,
  statusPanel,
  stockfish,
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
