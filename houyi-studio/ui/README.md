# HouYi Console UI

Frontend for HouYi Agent execution visualization and control.

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **React Flow** - DAG visualization
- **Zustand** - State management
- **TailwindCSS** - Styling
- **WebSocket** - Real-time communication

## Development

### Install Dependencies

```bash
npm install
```

### Run Development Server

```bash
npm run dev
```

The UI will be available at http://localhost:3000

### Build for Production

```bash
npm run build
```

### Type Check

```bash
npm run type-check
```

## Project Structure

```
src/
├── components/       # React components
│   ├── nodes/       # Custom node components
│   └── DAGCanvas.tsx
├── stores/          # Zustand stores
├── types/           # TypeScript types
├── utils/           # Utilities (WebSocket client)
├── App.tsx          # Main app component
└── main.tsx         # Entry point
```

## Backend Connection

The UI connects to the backend WebSocket server at `ws://localhost:8000/ws/session/{session_id}`.

Make sure the backend server is running before starting the UI.
