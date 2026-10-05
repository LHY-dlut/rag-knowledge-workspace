import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: { proxy: { '/api': { target: process.env.VITE_API_TARGET || 'http://127.0.0.1:8000', changeOrigin: true } } },
  build: { rolldownOptions: { output: { manualChunks(id) {
    if (id.includes('node_modules/element-plus')) return 'ui'
    if (id.includes('node_modules/echarts') || id.includes('node_modules/zrender')) return 'charts'
  } } } },
})
