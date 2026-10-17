import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// base 保持 '/'：nginx 把 frontend/dist 挂在站点根，try_files 已配好
// history 模式回退（deploy/nginx/conf.d/imcs.conf），无需子路径。
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: 5173,
    // 开发期同样走 /api 同源代理，且不 rewrite 掉前缀：
    // nginx 生产侧是原样透传到 api:8010，后端路由前缀本身就是 /api/v1。
    // 两边行为一致，切生产时 baseURL 不用改。
    proxy: {
      '/api': {
        target: 'http://localhost:8010',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    // 产物直接进 nginx 挂载目录，因此清空是安全且必要的（占位页会被覆盖）
    emptyOutDir: true,
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        // 按依赖体积手动分包：echarts 与 naive-ui 都很大，
        // 混进主包会拖慢首屏，分开后能被浏览器长期缓存。
        manualChunks: {
          echarts: ['echarts'],
          naive: ['naive-ui'],
          vendor: ['vue', 'vue-router', 'pinia', 'axios'],
        },
      },
    },
  },
})
