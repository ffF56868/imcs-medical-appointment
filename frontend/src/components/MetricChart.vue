<script setup lang="ts">
/**
 * 轻量 ECharts 折线封装：自适应容器宽、数据驱动重绘、卸载销毁。
 * 只做监控曲线这一个场景，不做通用图表库。
 */
import { onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue'
import * as echarts from 'echarts'


/** 图表系列：名称 + [x 标签, 数值] 点列 */
interface ChartSeries {
  name: string
  points: Array<[string, number]>
}

const props = defineProps<{
  series: ChartSeries[]
  height?: number
  /** y 轴刻度格式化；缺省原样显示 */
  yAxisFormat?: (v: number) => string
}>()

const el = ref<HTMLDivElement>()
const chart = shallowRef<echarts.ECharts>()

function render() {
  if (!chart.value) return
  chart.value.setOption({
    grid: { left: 56, right: 16, top: 30, bottom: 28 },
    tooltip: { trigger: 'axis' },
    legend: { top: 2, data: props.series.map((s) => s.name), textStyle: { fontSize: 12 } },
    xAxis: {
      type: 'category',
      data: props.series[0]?.points.map((p) => p[0]) ?? [],
      axisLabel: { fontSize: 11, color: '#6b7280' },
      axisLine: { lineStyle: { color: '#d1d5db' } },
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        fontSize: 11,
        color: '#6b7280',
        formatter: props.yAxisFormat ? (_: string, v: number) => props.yAxisFormat!(v) : undefined,
      },
      splitLine: { lineStyle: { color: '#e5e7eb' } },
    },
    series: props.series.map((s) => ({
      name: s.name,
      type: 'line' as const,
      data: s.points.map((p) => p[1]),
      showSymbol: false,
      smooth: false,
      lineStyle: { width: 1.5 },
      emphasis: { focus: 'series' as const },
    })),
  })
}

const resize = () => chart.value?.resize()

onMounted(() => {
  if (el.value) chart.value = echarts.init(el.value)
  render()
  window.addEventListener('resize', resize)
})
watch(() => [props.series, props.yAxisFormat], render, { deep: true })
onBeforeUnmount(() => {
  window.removeEventListener('resize', resize)
  chart.value?.dispose()
})
</script>

<template>
  <div ref="el" :style="{ width: '100%', height: `${height ?? 260}px` }" />
</template>
