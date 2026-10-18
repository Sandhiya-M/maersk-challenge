import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/shipments': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/port-events': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/ports': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/routes': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/pipeline': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/data-quality': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/predict': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
