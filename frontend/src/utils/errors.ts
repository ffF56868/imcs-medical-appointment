/**
 * 后端错误码 → 用户可读中文提示。
 *
 * 后端失败体是 { detail: { code, message } }，message 已是中文，但偏技术
 * （如「缺少权限：health_qa:ask」）。这里给常见 code 配更友好的话术，
 * 未命中的回退到后端 message，再回退到通用提示——三层兜底，
 * 保证任何错误都有人话可显示，不会出现空白 toast。
 */

export const ERROR_MESSAGES: Record<string, string> = {
  // 认证
  auth_invalid_credentials: '账号或密码不正确',
  auth_token_expired: '登录已过期，请重新登录',
  auth_token_invalid: '登录凭证无效，请重新登录',
  auth_account_locked: '账号已被临时锁定，请稍后再试',
  auth_forbidden: '当前角色无权执行该操作',

  // 号源与预约
  appointment_slot_full: '该时段号源已挂满，请选择其他时段',
  appointment_slot_conflict: '号源状态已变更，请刷新后重试',
  appointment_duplicate: '你在该时段已有预约',
  schedule_not_open: '该排班未开放预约',

  // 状态机
  state_transition_invalid: '当前状态不允许该操作',

  // 处方安全
  prescription_drug_interaction: '处方存在药物相互作用风险',
  prescription_allergy_conflict: '处方药物与患者过敏史冲突',

  // AI
  llm_all_providers_failed: 'AI 服务暂时不可用，请稍后再试',
  llm_token_budget_exceeded: 'AI 调用额度已用完',
  llm_content_blocked: '内容未通过安全审核',
  llm_structured_output_invalid: 'AI 返回格式异常，请重试',
  ai_task_not_found: 'AI 任务不存在或已过期',

  // 任务
  task_already_running: '已有同类操作正在进行，请等待完成',
  task_queue_unavailable: '任务队列暂时不可用，请稍后重试',

  // 检索
  retrieval_no_result: '未检索到相关资料',

  // 通用
  resource_not_found: '请求的资源不存在',
  internal_error: '服务器开小差了，请稍后再试',
  stream_timeout: '回答生成超时，请重新提问',
}

const FALLBACK = '操作失败，请稍后再试'

/** 从后端错误体解析出 { code, message, detail }，兼容 detail 为字符串或对象两种形态。 */
export function parseApiError(data: unknown): { code: string; message: string; detail: unknown } {
  const detail = (data as { detail?: unknown })?.detail
  if (typeof detail === 'string') {
    return { code: '', message: detail || FALLBACK, detail: null }
  }
  if (detail && typeof detail === 'object') {
    const d = detail as { code?: string; message?: string }
    const code = d.code ?? ''
    const message = ERROR_MESSAGES[code] ?? d.message ?? FALLBACK
    // 原样带出 detail 对象：处方拦截等业务错误的 findings 藏在这里，
    // 只留 code/message 的话医生端就没法展示「具体哪条安全规则命中」。
    return { code, message, detail }
  }
  return { code: '', message: FALLBACK, detail: null }
}
