import { useQuery } from '@tanstack/react-query'
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, Tooltip, ResponsiveContainer, Legend
} from 'recharts'
import { Users, TrendingUp, AlertTriangle, Brain, Stethoscope } from 'lucide-react'
import { analyticsApi } from '../services/api'
import type { DashboardData } from '../types'
import { StatCard, Spinner, PageHeader, SectionCard } from '../components/shared'
import { formatScore, riskChartColors } from '../utils/helpers'

const PIE_COLORS = ['#10b981', '#f59e0b', '#f97316', '#ef4444']

export default function DashboardPage() {
  const { data, isLoading, isError } = useQuery<DashboardData>({
    queryKey: ['dashboard'],
    queryFn: async () => {
      const res = await analyticsApi.dashboard()
      return res.data
    },
    refetchInterval: 60_000,
  })

  if (isLoading) return <Spinner label="Loading dashboard…" />
  if (isError || !data) return (
    <div className="card p-8 text-center text-slate-400">
      <AlertTriangle className="w-8 h-8 mx-auto mb-2 text-amber-400" />
      <p className="text-sm">Could not load dashboard. Make sure the backend is running.</p>
    </div>
  )

  const { stats, risk_distribution, risk_trend, top_risk_factors_global } = data

  const pieData = [
    { name: 'Low', value: risk_distribution.low },
    { name: 'Medium', value: risk_distribution.medium },
    { name: 'High', value: risk_distribution.high },
    { name: 'Critical', value: risk_distribution.critical },
  ]

  const trendData = risk_trend.map(p => ({
    date: p.date,
    score: Math.round(p.value * 100),
  }))

  const factorData = top_risk_factors_global.slice(0, 8).map(f => ({
    name: f.feature.length > 18 ? f.feature.slice(0, 16) + '…' : f.feature,
    shap: +(f.avg_shap * 100).toFixed(2),
  }))

  return (
    <div className="space-y-6">
      <PageHeader
        title="Dashboard"
        subtitle="Real-time overview of patient health analytics"
      />

      {/* KPI Row */}
      <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
        <StatCard
          label="Total Patients"
          value={stats.total_patients}
          icon={<Users className="w-5 h-5 text-indigo-600" />}
          iconBg="bg-indigo-50"
        />
        <StatCard
          label="Records This Month"
          value={stats.records_this_month}
          icon={<TrendingUp className="w-5 h-5 text-blue-600" />}
          iconBg="bg-blue-50"
        />
        <StatCard
          label="High Risk Patients"
          value={stats.high_risk_count}
          icon={<AlertTriangle className="w-5 h-5 text-orange-600" />}
          iconBg="bg-orange-50"
        />
        <StatCard
          label="Avg Risk Score"
          value={formatScore(stats.avg_risk_score)}
          icon={<Brain className="w-5 h-5 text-purple-600" />}
          iconBg="bg-purple-50"
        />
        <StatCard
          label="Triages Today"
          value={stats.triage_sessions_today}
          icon={<Stethoscope className="w-5 h-5 text-teal-600" />}
          iconBg="bg-teal-50"
        />
      </div>

      {/* Charts row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Risk trend */}
        <div className="lg:col-span-2">
          <SectionCard title="30-Day Risk Score Trend">
            {trendData.length > 0 ? (
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={trendData} margin={{ top: 5, right: 10, bottom: 0, left: -10 }}>
                  <defs>
                    <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#6366f1" stopOpacity={0.2} />
                      <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} axisLine={false} domain={[0, 100]} unit="%" />
                  <Tooltip
                    contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e2e8f0' }}
                    formatter={(v: number) => [`${v}%`, 'Avg Risk']}
                  />
                  <Area type="monotone" dataKey="score" stroke="#6366f1" strokeWidth={2} fill="url(#riskGrad)" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[220px] flex items-center justify-center text-slate-400 text-sm">
                No trend data yet — run some predictions first
              </div>
            )}
          </SectionCard>
        </div>

        {/* Risk distribution pie */}
        <SectionCard title="Risk Distribution">
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%" cy="45%"
                innerRadius={55} outerRadius={80}
                paddingAngle={3}
                dataKey="value"
              >
                {pieData.map((_, i) => (
                  <Cell key={i} fill={PIE_COLORS[i]} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => [v, 'Patients']} contentStyle={{ fontSize: 12, borderRadius: 8 }} />
              <Legend iconSize={10} iconType="circle" wrapperStyle={{ fontSize: 12 }} />
            </PieChart>
          </ResponsiveContainer>
        </SectionCard>
      </div>

      {/* Charts row 2 */}
      <SectionCard title="Top Global Risk Factors (Avg SHAP Impact)">
        {factorData.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={factorData} layout="vertical" margin={{ top: 0, right: 20, bottom: 0, left: 0 }}>
              <XAxis type="number" tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} axisLine={false} />
              <YAxis dataKey="name" type="category" width={140} tick={{ fontSize: 11, fill: '#64748b' }} tickLine={false} axisLine={false} />
              <Tooltip
                contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e2e8f0' }}
                formatter={(v: number) => [`${v}`, 'Avg SHAP']}
              />
              <Bar dataKey="shap" fill="#6366f1" radius={[0, 4, 4, 0]} maxBarSize={14} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-[200px] flex items-center justify-center text-slate-400 text-sm">
            No prediction data yet
          </div>
        )}
      </SectionCard>
    </div>
  )
}
