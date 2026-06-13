import type { RiskLevel, UrgencyLevel } from '../types'

export const riskColors: Record<RiskLevel, { bg: string; text: string; border: string; dot: string }> = {
  low:      { bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200', dot: 'bg-emerald-500' },
  medium:   { bg: 'bg-amber-50',   text: 'text-amber-700',   border: 'border-amber-200',   dot: 'bg-amber-500'   },
  high:     { bg: 'bg-orange-50',  text: 'text-orange-700',  border: 'border-orange-200',  dot: 'bg-orange-500'  },
  critical: { bg: 'bg-red-50',     text: 'text-red-700',     border: 'border-red-200',     dot: 'bg-red-500'     },
}

export const urgencyColors: Record<UrgencyLevel, { bg: string; text: string; label: string }> = {
  routine:   { bg: 'bg-slate-100',   text: 'text-slate-600',   label: 'Routine'   },
  soon:      { bg: 'bg-blue-50',     text: 'text-blue-700',    label: 'See Soon'  },
  urgent:    { bg: 'bg-orange-50',   text: 'text-orange-700',  label: 'Urgent'    },
  emergency: { bg: 'bg-red-50',      text: 'text-red-700',     label: 'Emergency' },
}

export const riskChartColors: Record<RiskLevel, string> = {
  low: '#10b981', medium: '#f59e0b', high: '#f97316', critical: '#ef4444',
}

export function riskBadgeClass(level: RiskLevel) {
  return `badge-${level}`
}

export function formatScore(score: number) {
  return (score * 100).toFixed(1) + '%'
}

export function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString('en-IN', {
    day: '2-digit', month: 'short', year: 'numeric',
  })
}

export function formatDateTime(iso: string) {
  return new Date(iso).toLocaleString('en-IN', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

export function capitalize(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1)
}

export function getRiskLabel(score: number): RiskLevel {
  if (score < 0.25) return 'low'
  if (score < 0.50) return 'medium'
  if (score < 0.75) return 'high'
  return 'critical'
}
