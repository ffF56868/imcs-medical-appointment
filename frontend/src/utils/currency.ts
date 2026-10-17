/**
 * 金额工具。全站展示金额**必须**经过这里，不允许在组件里手写 /100。
 *
 * 后端有两套口径（这是最容易出错的点，review 必查）：
 *   - int「分」：registration_fee / total_amount / paid_amount / ChargeItem.amount
 *   - float「元」：处方明细 unit_price
 * fenToYuan 只处理「分」；unit_price 本来就是元，直接 formatYuan。
 */

/** 分 → 元（数值）。用整数运算避免浮点误差：1050 分 = 10.5 元。 */
export function fenToYuan(fen: number | null | undefined): number {
  if (fen == null || Number.isNaN(fen)) return 0
  return Math.round(fen) / 100
}

/** 元数值格式化为「¥12.34」。用于 unit_price 这类本就是元的字段。
 * 入参兼容字符串：后端处方明细的 unit_price/quantity 是 Numeric，
 * Pydantic 序列化成 JSON 后是字符串，直接 toFixed 会抛。 */
export function formatYuan(yuan: number | string | null | undefined): string {
  const n = typeof yuan === 'string' ? Number(yuan) : yuan
  return `¥${(Number.isFinite(n as number) ? (n as number) : 0).toFixed(2)}`
}

/** 分 → 「¥12.34」。用于 registration_fee / total_amount 等 int 分字段。 */
export function formatFen(fen: number | null | undefined): string {
  return formatYuan(fenToYuan(fen))
}

/** 元 → 分（提交处方 unit_price 以外的金额时用）。四舍五入到整分。 */
export function yuanToFen(yuan: number | null | undefined): number {
  if (yuan == null || Number.isNaN(yuan)) return 0
  return Math.round(yuan * 100)
}
