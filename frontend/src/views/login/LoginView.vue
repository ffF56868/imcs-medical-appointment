<script setup lang="ts">
/**
 * 登录页。左侧品牌信息区（深蓝纯色，无渐变无插图），右侧登录表单。
 * 演示账号快捷填充：六角色一键填入，方便评审与联调。
 */
import { ref, reactive } from 'vue'
import {
  NForm, NFormItem, NInput, NButton, NCard, useMessage,
} from 'naive-ui'
import type { FormInst, FormRules } from 'naive-ui'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { ApiError } from '@/api/client'

const router = useRouter()
const route = useRoute()
const auth = useAuthStore()
const message = useMessage()

const formRef = ref<FormInst | null>(null)
const loading = ref(false)
const model = reactive({ account: '', password: '' })

const rules: FormRules = {
  account: { required: true, message: '请输入手机号 / 邮箱 / 用户名', trigger: 'blur' },
  password: { required: true, message: '请输入密码', trigger: 'blur' },
}

// 演示账号（种子数据，密码统一 Imcs2026demo）
const demoAccounts = [
  { role: 'patient', label: '患者', account: 'patient01' },
  { role: 'doctor', label: '医生', account: 'doctor01' },
  { role: 'pharmacist', label: '药师', account: 'pharmacist01' },
  { role: 'cashier', label: '收费员', account: 'cashier01' },
  { role: 'lab_tech', label: '检验科', account: 'lab01' },
  { role: 'admin', label: '管理员', account: 'admin' },
]
const DEMO_PASSWORD = 'Imcs2026demo'

function fillDemo(account: string) {
  model.account = account
  model.password = DEMO_PASSWORD
}

async function handleLogin() {
  try {
    await formRef.value?.validate()
  } catch {
    return
  }
  loading.value = true
  try {
    const user = await auth.login(model.account.trim(), model.password)
    message.success(`欢迎，${user.real_name || user.username}`)
    const redirect = (route.query.redirect as string) || ''
    // 登录成功后按角色回各自首页；带 redirect 则优先回跳
    await router.replace(redirect || { name: 'home' })
  } catch (err) {
    const msg = err instanceof ApiError ? err.message : '登录失败，请稍后再试'
    message.error(msg)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login">
    <div class="login__brand">
      <div class="login__brand-inner">
        <h1 class="login__brand-title">IMCS</h1>
        <p class="login__brand-sub">智能医疗预约诊疗系统</p>
        <p class="login__brand-desc">
          覆盖分诊、预约、接诊、检查、处方、收费、发药与 AI 监控评测的一体化平台
        </p>
      </div>
    </div>
    <div class="login__panel">
      <n-card class="login__card" :bordered="false">
        <h2 class="login__title">登录</h2>
        <p class="login__hint">请使用系统账号登录</p>
        <n-form
          ref="formRef"
          :model="model"
          :rules="rules"
          label-placement="top"
          @keyup.enter="handleLogin"
        >
          <n-form-item label="账号" path="account">
            <n-input
              v-model:value="model.account"
              placeholder="手机号 / 邮箱 / 用户名"
              size="large"
              clearable
            />
          </n-form-item>
          <n-form-item label="密码" path="password">
            <n-input
              v-model:value="model.password"
              type="password"
              placeholder="请输入密码"
              size="large"
              show-password-on="click"
            />
          </n-form-item>
          <n-button
            type="primary"
            size="large"
            block
            :loading="loading"
            @click="handleLogin"
          >
            登录
          </n-button>
        </n-form>

        <div class="login__demo">
          <div class="login__demo-label">演示账号（点击填充）</div>
          <div class="login__demo-tags">
            <n-button
              v-for="d in demoAccounts"
              :key="d.account"
              size="tiny"
              secondary
              @click="fillDemo(d.account)"
            >
              {{ d.label }} · {{ d.account }}
            </n-button>
          </div>
        </div>
      </n-card>
    </div>
  </div>
</template>

<style scoped>
.login {
  display: flex;
  height: 100%;
  background: var(--gray-3);
}
.login__brand {
  flex: 1;
  background: var(--brand-7);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--space-12);
}
.login__brand-inner {
  max-width: 420px;
  color: #fff;
}
.login__brand-title {
  font-size: 44px;
  font-weight: 700;
  letter-spacing: 2px;
  margin: 0;
}
.login__brand-sub {
  font-size: var(--font-xl);
  margin: var(--space-3) 0 var(--space-6);
  opacity: 0.92;
}
.login__brand-desc {
  font-size: var(--font-base);
  line-height: 1.8;
  opacity: 0.72;
  margin: 0;
}
.login__panel {
  width: 460px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--space-8);
}
.login__card {
  width: 100%;
}
.login__title {
  font-size: 22px;
  font-weight: 600;
  color: var(--gray-9);
  margin: 0 0 var(--space-1);
}
.login__hint {
  font-size: var(--font-sm);
  color: var(--gray-7);
  margin: 0 0 var(--space-6);
}
.login__demo {
  margin-top: var(--space-6);
  padding-top: var(--space-4);
  border-top: 1px solid var(--border-color-light);
}
.login__demo-label {
  font-size: var(--font-xs);
  color: var(--gray-7);
  margin-bottom: var(--space-2);
}
.login__demo-tags {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
}
@media (max-width: 900px) {
  .login__brand { display: none; }
  .login__panel { width: 100%; }
}
</style>
