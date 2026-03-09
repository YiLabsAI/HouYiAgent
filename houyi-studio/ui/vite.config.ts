import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const backendPort = process.env.HOUYI_PORT || process.env.HOUYI_E2E_BACKEND_PORT || '8000'
  const uiPort = parseInt(process.env.HOUYI_UI_PORT || process.env.HOUYI_E2E_UI_PORT || '3000', 10)

  return {
    plugins: [react()],
    optimizeDeps: {
      exclude: ['playwright', 'playwright-core', 'chromium-bidi', 'fsevents'],
    },
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            vendor: ['react', 'react-dom', 'zustand'],
            reactflow: ['reactflow'],
          },
        },
      },
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      port: uiPort,
      hmr: {
        overlay: true,
      },
      watch: {
        usePolling: true,
      },
      headers: {
        'Cache-Control': 'no-store, no-cache, must-revalidate, proxy-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
      },
      proxy: {
        '/api': {
          target: `http://localhost:${backendPort}`,
          changeOrigin: true,
        },
        '/ws': {
          target: `ws://localhost:${backendPort}`,
          ws: true,
        },
      },
    },
  }
})
