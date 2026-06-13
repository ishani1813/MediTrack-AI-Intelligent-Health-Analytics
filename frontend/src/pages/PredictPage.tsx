import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine
} from 'recharts'
import { Brain, Zap, Info } from 'lucide-react'
import { predictApi } from '../services/api'
import type { Prediction, RiskLevel } from '../types'
import { RiskBadge, RiskScoreRing, PageHeader, SectionCard } from '../components/shared'

const DEFAULT_VITALS = {
  patient_id: 1,
  age: 45,
  blood_pressure_systolic: 135,
  blood_pressure_diastolic: 88,
  heart_rate: 78,
  blood_glucose: 115,
  bmi: 27.5,
  cholesterol_total: 210,
  cholesterol_hdl: 48,
  cholesterol_ldl: 135,
  hemoglobin: 13.2,
  oxygen_saturation: 97,
}

const FIELD_META: Record<string, { label: string; unit: string; normal: string }> = {
  age: { label: 'Age', unit: 'yrs', normal: '' },
  blood_pressure_systolic: { label: 'Systolic BP', unit: 'mmHg', normal: '<120' },
  blood_pressure_diastolic: { label: 'Diastolic BP', unit: 'mmHg', normal: '<80' },
  heart_rate: { label: 'Heart Rate', unit: 'bpm', normal: '60–100' },
  blood_glucose: { label: 'Blood Glucose', unit: 'mg/dL', normal: '70–100' },
  bmi: { label: 'BMI', unit: 'kg/m²', normal: '18.5–24.9' },
  cholesterol_total: { label: 'Total Cholesterol', unit: 'mg/dL', normal: '<200' },
  cholesterol_hdl: { label: 'HDL', unit: 'mg/dL', normal: '>40' },
  cholesterol_ldl: { label: 'LDL', unit: 'mg/dL', normal: '<100' },
  hemoglobin: { label: 'Hemoglobin', unit: 'g/dL', normal: '12–17' },
  oxygen_saturation: { label: 'O₂ Saturation', unit: '%', normal: '>95' },
}

export default function PredictPage() {
  const [form, setForm] = useState(DEFAULT_VITALS)
  const [result, setResult] = useState<Prediction | null>(null)

  const mutation = useMutation({
    mutationFn: () => predictApi.predict(form),
    onSuccess: (res) => setResult(res.data),
  })

  const shapData = result?.top_risk_factors.map(f => ({
    name: f.feature.length > 20 ? f.feature.slice(0, 18) + '…' : f.feature,
    value: f.shap_value,
    fill: f.impact === 'increases_risk' ? '#ef4444' : '#10b981',
    impact: f.impact,
    rawValue: f.value,
  })) || []

  const fields = Object.entries(form).filter(([k]) => k !== 'patient_id')

  return (
    <div className="space-y-6">
      <PageHeader
        title="Risk Prediction"
        subtitle="XGBoost + Random Forest ensemble with SHAP explainability"
      />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Input form */}
        <div className="card p-5">
          <div className="flex items-center gap-2 mb-5">
            <div className="w-7 h-7 bg-indigo-50 rounded-lg flex items-center justify-center">
              <Brain className="w-4 h-4 text-indigo-600" />
            </div>
            <h2 className="text-sm font-semibold text-slate-700">Patient Vitals Input</h2>
          </div>

          <div className="mb-3">
            <label className="label">Patient ID</label>
            <input className="input" type="number" value={form.patient_id}
              onChange={e => setForm(f => ({ ...f, patient_id: +e.target.value }))} />
          </div>

          <div className="grid grid-cols-2 gap-3">
            {fields.map(([key, val]) => {
              const meta = FIELD_META[key]
              return (
                <div key={key}>
                  <label className="label flex items-center justify-between">
                    <span>{meta?.label || key}</span>
                    {meta?.normal && <span className="text-slate-300 font-normal text-[10px]">Normal: {meta.normal}</span>}
                  </label>
                  <div className="relative">
                    <input className="input pr-10" type="number" step="0.1" value={val as number}
                      onChange={e => setForm(f => ({ ...f, [key]: +e.target.value }))} />
                    <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-slate-400">{meta?.unit}</span>
                  </div>
                </div>
              )
            })}
          </div>

          <button
            className="btn-primary w-full mt-5 flex items-center justify-center gap-2"
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
          >
            <Zap className="w-4 h-4" />
            {mutation.isPending ? 'Running prediction…' : 'Predict Risk'}
          </button>

          {mutation.isError && (
            <p className="text-xs text-red-500 mt-2 bg-red-50 p-2 rounded-lg">
              Prediction failed — check backend connection
            </p>
          )}
        </div>

        {/* Results */}
        {result ? (
          <div className="space-y-4">
            {/* Score card */}
            <div className="card p-5">
              <div className="flex items-center gap-6">
                <RiskScoreRing score={result.risk_score} level={result.risk_level as RiskLevel} />
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <RiskBadge level={result.risk_level as RiskLevel} />
                    {result.cached && (
                      <span className="text-xs text-slate-400 bg-slate-100 px-2 py-0.5 rounded-full">cached</span>
                    )}
                  </div>
                  <p className="text-xs text-slate-500 mb-2">
                    Confidence: <span className="font-medium text-slate-700">{(result.confidence * 100).toFixed(1)}%</span>
                    &nbsp;·&nbsp;Model: <span className="font-medium">{result.model_version}</span>
                  </p>
                  <p className="text-xs text-slate-600 leading-relaxed bg-slate-50 rounded-lg px-3 py-2">
                    {result.recommendation}
                  </p>
                </div>
              </div>
            </div>

            {/* SHAP waterfall */}
            <SectionCard title="SHAP Feature Importance">
              <p className="text-xs text-slate-400 mb-3 flex items-center gap-1">
                <Info className="w-3 h-3" />
                Red = increases risk &nbsp;|&nbsp; Green = decreases risk
              </p>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={shapData} layout="vertical" margin={{ top: 0, right: 20, bottom: 0, left: 0 }}>
                  <XAxis type="number" tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} axisLine={false} />
                  <YAxis dataKey="name" type="category" width={148} tick={{ fontSize: 11, fill: '#64748b' }} tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e2e8f0' }}
                    formatter={(v: number, _: string, entry: any) => [
                      `SHAP: ${v.toFixed(4)} · Value: ${entry.payload.rawValue}`,
                      entry.payload.impact === 'increases_risk' ? '↑ Increases Risk' : '↓ Decreases Risk',
                    ]}
                  />
                  <ReferenceLine x={0} stroke="#e2e8f0" />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={16}>
                    {shapData.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </SectionCard>

            {/* Top factors list */}
            <SectionCard title="Top Risk Drivers">
              <div className="space-y-2">
                {result.top_risk_factors.map((f, i) => (
                  <div key={i} className="flex items-center gap-3 text-sm">
                    <span className={`w-2 h-2 rounded-full shrink-0 ${f.impact === 'increases_risk' ? 'bg-red-400' : 'bg-emerald-400'}`} />
                    <span className="flex-1 text-slate-700">{f.feature}</span>
                    <span className="text-slate-400 text-xs tabular-nums">val: {f.value}</span>
                    <span className={`text-xs font-mono tabular-nums ${f.impact === 'increases_risk' ? 'text-red-500' : 'text-emerald-600'}`}>
                      {f.shap_value > 0 ? '+' : ''}{f.shap_value.toFixed(4)}
                    </span>
                  </div>
                ))}
              </div>
            </SectionCard>
          </div>
        ) : (
          <div className="card flex flex-col items-center justify-center py-20 text-center">
            <div className="w-16 h-16 bg-indigo-50 rounded-2xl flex items-center justify-center mb-4">
              <Brain className="w-8 h-8 text-indigo-400" />
            </div>
            <p className="text-sm font-medium text-slate-600">Prediction results appear here</p>
            <p className="text-xs text-slate-400 mt-1">Fill in vitals and click Predict Risk</p>
          </div>
        )}
      </div>
    </div>
  )
}
