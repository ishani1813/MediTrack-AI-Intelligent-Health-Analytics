import type { RiskLevel, UrgencyLevel } from '../../types'
import { riskColors, urgencyColors, formatScore } from '../../utils/helpers'
import { Loader2 } from 'lucide-react'

// ── RiskBadge ─────────────────────────────────────────────────────────────────
export function RiskBadge({ level }: { level: RiskLevel }) {
  const c = riskColors[level]
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium ${c.bg} ${c.text}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${c.dot}`} />
      {level.charAt(0).toUpperCase() + level.slice(1)}
    </span>
  )
}

// ── UrgencyBadge ──────────────────────────────────────────────────────────────
export function UrgencyBadge({ level }: { level: UrgencyLevel }) {
  const c = urgencyColors[level]
  return (
    <span className={`inline-flex px-2.5 py-0.5 rounded-full text-xs font-semibold ${c.bg} ${c.text}`}>
      {c.label}
    </span>
  )
}

// ── StatCard ──────────────────────────────────────────────────────────────────
interface StatCardProps {
  label: string
  value: string | number
  sub?: string
  icon: React.ReactNode
  iconBg?: string
  trend?: { value: number; label: string }
}
export function StatCard({ label, value, sub, icon, iconBg = 'bg-indigo-50', trend }: StatCardProps) {
  return (
    <div className="card p-5">
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">{label}</p>
          <p className="text-2xl font-bold text-slate-800 mt-1">{value}</p>
          {sub && <p className="text-xs text-slate-400 mt-0.5">{sub}</p>}
          {trend && (
            <p className={`text-xs mt-1 font-medium ${trend.value >= 0 ? 'text-emerald-600' : 'text-red-500'}`}>
              {trend.value >= 0 ? '↑' : '↓'} {Math.abs(trend.value)}% {trend.label}
            </p>
          )}
        </div>
        <div className={`w-10 h-10 rounded-xl ${iconBg} flex items-center justify-center shrink-0 ml-3`}>
          {icon}
        </div>
      </div>
    </div>
  )
}

// ── Spinner ───────────────────────────────────────────────────────────────────
export function Spinner({ size = 'md', label = 'Loading...' }: { size?: 'sm' | 'md' | 'lg'; label?: string }) {
  const sz = { sm: 'w-4 h-4', md: 'w-6 h-6', lg: 'w-8 h-8' }[size]
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-10 text-slate-400">
      <Loader2 className={`${sz} animate-spin`} />
      <p className="text-sm">{label}</p>
    </div>
  )
}

// ── EmptyState ────────────────────────────────────────────────────────────────
export function EmptyState({ icon, title, description, action }: {
  icon: React.ReactNode
  title: string
  description?: string
  action?: React.ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center px-4">
      <div className="w-14 h-14 bg-slate-100 rounded-full flex items-center justify-center text-slate-400 mb-4">
        {icon}
      </div>
      <h3 className="text-sm font-semibold text-slate-700">{title}</h3>
      {description && <p className="text-sm text-slate-400 mt-1 max-w-xs">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

// ── PageHeader ────────────────────────────────────────────────────────────────
export function PageHeader({ title, subtitle, action }: {
  title: string
  subtitle?: string
  action?: React.ReactNode
}) {
  return (
    <div className="flex items-start justify-between mb-6 gap-4 flex-wrap">
      <div>
        <h1 className="text-xl font-bold text-slate-800">{title}</h1>
        {subtitle && <p className="text-sm text-slate-500 mt-0.5">{subtitle}</p>}
      </div>
      {action && <div>{action}</div>}
    </div>
  )
}

// ── RiskScoreRing ─────────────────────────────────────────────────────────────
export function RiskScoreRing({ score, level }: { score: number; level: RiskLevel }) {
  const pct = Math.round(score * 100)
  const radius = 36
  const circ = 2 * Math.PI * radius
  const dash = (pct / 100) * circ
  const color = { low: '#10b981', medium: '#f59e0b', high: '#f97316', critical: '#ef4444' }[level]

  return (
    <div className="relative w-24 h-24 flex items-center justify-center">
      <svg className="absolute inset-0 -rotate-90" width="96" height="96" viewBox="0 0 96 96">
        <circle cx="48" cy="48" r={radius} fill="none" stroke="#e2e8f0" strokeWidth="8" />
        <circle
          cx="48" cy="48" r={radius} fill="none"
          stroke={color} strokeWidth="8"
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
        />
      </svg>
      <div className="text-center">
        <p className="text-lg font-bold text-slate-800">{pct}%</p>
        <p className="text-[10px] text-slate-400 capitalize">{level}</p>
      </div>
    </div>
  )
}

// ── SectionCard ───────────────────────────────────────────────────────────────
export function SectionCard({ title, children, action }: {
  title: string
  children: React.ReactNode
  action?: React.ReactNode
}) {
  return (
    <div className="card">
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-slate-100">
        <h2 className="text-sm font-semibold text-slate-700">{title}</h2>
        {action}
      </div>
      <div className="p-5">{children}</div>
    </div>
  )
}
