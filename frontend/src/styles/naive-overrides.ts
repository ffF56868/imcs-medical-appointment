import type { GlobalThemeOverrides } from 'naive-ui'

/**
 * Naive UI 主题覆盖。把 tokens.css 里的临床克制风 token 翻译成 Naive 的主题变量。
 *
 * **为什么用 themeOverrides 而不是改 CSS 变量**：Naive 组件的内部样式（按钮 hover、
 * 表格边框、tag 配色）由它自己的 theme 变量驱动，直接写 CSS 覆盖要 !important
 * 硬怼且极易漏。这里统一在 NConfigProvider 注入，组件天然跟随。
 *
 * 值与 tokens.css 保持一致（手动同步，两处都改）：CSS 变量供自定义样式用，
 * 这里供 Naive 组件用。若两者对不上，说明有人只改了一处 —— review 必查。
 */
export const themeOverrides: GlobalThemeOverrides = {
  common: {
    primaryColor: '#1b5e7b',
    primaryColorHover: '#267a9e',
    primaryColorPressed: '#13475d',
    primaryColorSuppl: '#267a9e',
    infoColor: '#1565c0',
    successColor: '#2e7d32',
    warningColor: '#c77800',
    errorColor: '#c62828',
    // 中性色阶：Naive 用 bodyColor/baseColor 等命名，映射到我们的灰阶
    bodyColor: '#ffffff',
    baseColor: '#ffffff',
    cardColor: '#ffffff',
    modalColor: '#ffffff',
    popoverColor: '#ffffff',
    tableColor: '#ffffff',
    inputColor: '#ffffff',
    borderColor: '#dcdfe6',
    dividerColor: '#e8e8e8',
    hoverColor: '#f5f7fa',
    tableHeaderColor: '#f5f7fa',
    textColorBase: '#1a1a1a',
    textColor1: '#1a1a1a',
    textColor2: '#606266',
    textColor3: '#909399',
    placeholderColor: '#909399',
    // 圆角只有两档
    borderRadius: '4px',
    borderRadiusSmall: '2px',
    fontSize: '14px',
    fontSizeSmall: '13px',
    fontSizeMedium: '14px',
    fontSizeLarge: '16px',
    // 紧凑：医疗系统一屏要看很多数据
    heightMedium: '34px',
    heightSmall: '28px',
    heightLarge: '40px',
  },
  Button: {
    borderRadiusMedium: '4px',
    borderRadiusSmall: '2px',
    fontWeight: '400',
  },
  DataTable: {
    thColor: '#f5f7fa',
    thTextColor: '#1a1a1a',
    thFontWeight: '600',
    borderColor: '#e8e8e8',
    tdColorHover: '#f5f7fa',
    tdColorStriped: '#fafbfc',
    thPaddingSmall: '8px 12px',
    tdPaddingSmall: '8px 12px',
    fontSizeMedium: '13px',
  },
  Card: {
    paddingMedium: '16px',
    borderRadius: '4px',
    borderColor: '#e8e8e8',
  },
  Tag: {
    borderRadius: '2px',
    fontSizeMedium: '12px',
  },
  Input: {
    borderRadius: '4px',
  },
  Menu: {
    itemColorActive: '#ebf4f7',
    itemColorActiveHover: '#ebf4f7',
    itemTextColorActive: '#1b5e7b',
    itemIconColorActive: '#1b5e7b',
    borderRadius: '4px',
  },
}
