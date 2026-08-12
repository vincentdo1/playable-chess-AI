const isFile = window.location.protocol === 'file:';
const isLocalHost = ['localhost', '127.0.0.1'].includes(window.location.hostname);
const isGitHubPages = window.location.hostname.endsWith('github.io');
const repositoryBase = isGitHubPages ? '/playable-chess-AI' : '';
const assetBase = isFile ? '.' : repositoryBase;

export const config = {
  apiUrl: isFile || isLocalHost
    ? 'http://localhost:5000'
    : 'https://chess-ai-production-production.up.railway.app',
  piecesPath: `${assetBase}/pieces`,
  stockfishPath: `${assetBase}/stockfish.js`,
  magnus: {
    temperature: 0.0,
    valueWeight: 2.0,
    valueCandidates: 0,
  },
};
