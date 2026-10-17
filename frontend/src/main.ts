import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'
import { configureAuthHooks } from './api/client'
import { useAuthStore } from './stores/auth'
import './styles/base.css'

const app = createApp(App)
const pinia = createPinia()

app.use(pinia)
app.use(router)

// 把 auth store 的 token 访问器注入 axios client，打破 client↔store 循环依赖。
// 必须在 pinia 安装后、首次请求前调用。
const auth = useAuthStore(pinia)
configureAuthHooks({
  getToken: () => auth.accessToken,
  onUnauthorized: () => {
    auth.clearAll()
    // 用 router 而非 location.href，保留 SPA 状态并触发守卫记录 redirect
    const current = router.currentRoute.value
    if (current.name !== 'login') {
      router.replace({ name: 'login', query: { redirect: current.fullPath } })
    }
  },
  refresh: () => auth.tryRefresh(),
})

app.mount('#app')
