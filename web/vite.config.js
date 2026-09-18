import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 开发时把 /query /health 等 API 路径代理到后端；生产构建产物由 FastAPI 托管。
export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/query': 'http://localhost:8000',
      '/ingest': 'http://localhost:8000',
      '/documents': 'http://localhost:8000',
      '/doc/': 'http://localhost:8000',
      '/sessions': 'http://localhost:8000',
      '/session/': 'http://localhost:8000',
      '/agent': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
  build: { outDir: 'dist' },
})