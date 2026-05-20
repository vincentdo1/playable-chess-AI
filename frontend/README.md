# Frontend

The browser app is a static ES-module frontend. There is no build step yet; open it from the repo root with a local static server so module imports work.

```powershell
python -m http.server 8000
```

Then visit `http://localhost:8000`.

## Structure

- `src/main.js` wires the app together.
- `src/config.js` owns backend and asset paths.
- `src/api/` contains Flask API calls.
- `src/components/` contains DOM-facing UI controllers.
- `src/game/` contains chess game orchestration.
- `src/services/` contains external engines such as Stockfish.
- `src/styles/` contains app CSS.
