/**
 * 侧栏图标：用一个函数把 meta.icon 字符串映射成 Naive 图标组件。
 *
 * 只引入用到的图标（按需 import），避免把整个图标库打进包。
 * 图标风格统一用 Ionicons5 线性图标——细线条、无填充，符合临床克制风，
 * 不用彩色 emoji 或填充图标。
 */
import { h, type Component } from 'vue'
import { NIcon } from 'naive-ui'
import {
  HomeOutline,
  PulseOutline,
  ChatbubbleEllipsesOutline,
  CalendarOutline,
  DocumentTextOutline,
  ReceiptOutline,
  MedkitOutline,
  PeopleOutline,
  FlaskOutline,
  CashOutline,
  AnalyticsOutline,
  CheckmarkDoneOutline,
  LibraryOutline,
  ClipboardOutline,
} from '@vicons/ionicons5'

const ICON_MAP: Record<string, Component> = {
  home: HomeOutline,
  triage: PulseOutline,
  chat: ChatbubbleEllipsesOutline,
  calendar: CalendarOutline,
  record: DocumentTextOutline,
  report: ClipboardOutline,
  prescription: ReceiptOutline,
  bill: CashOutline,
  queue: PeopleOutline,
  pharmacy: MedkitOutline,
  lab: FlaskOutline,
  monitor: AnalyticsOutline,
  eval: CheckmarkDoneOutline,
  knowledge: LibraryOutline,
}

/** 返回一个渲染函数，供 Naive MenuOption.icon 使用。 */
export function SideMenuIcon(name: string) {
  const comp = ICON_MAP[name]
  if (!comp) return undefined
  return () => h(NIcon, null, { default: () => h(comp) })
}
