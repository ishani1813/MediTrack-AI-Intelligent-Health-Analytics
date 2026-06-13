import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts'
import { ArrowLeft, Plus, Brain, X, Activity } from 'lucide-react'
import { patientApi, predictApi } from '../services/api'
import type { Patient, HealthRecord, Prediction } from '../types'
import { RiskBadge, RiskScoreRing, Spinner, SectionCard } from '../components/shared'
import { formatDate, formatDateTime } from '../utils/helpers'

function AddRecordModal({ patientId, onClose }: { patientId: number; onClose: () => void }) {
  const qc = useQueryClient()
  const [form, setForm] = useState({
    blood_pressure_systolic: 120, blood_pressure_diastolic: 80,
    heart_rate: 72, blood_glucose: 95, bmi: 22,
    cholesterol_total: 180, cholesterol_hdl: 55, cholesterol_ldl: 100,
    hemoglobin: 14, temperature: 36.8, oxygen_saturation: 98, notes: '',
  })

  const mutation = useMutation({
    mutationFn: () => patientApi.addRecord(patientId, { ...form, patient_id: patientId }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['records', patientId] }); onClose() },
  })

  const fields: [string, keyof typeof form, string?][] = [
    ['Systolic BP', 'blood_pressure_systolic'],
    ['Diastolic BP', 'blood_pressure_diastolic'],
    ['Heart Rate', 'heart_rate'],
    ['Blood Glucose', 'blood_glucose'],
    ['BMI', 'bmi'],
    ['Total Cholesterol', 'cholesterol_total'],
    ['HDL', 'cholesterol_hdl'],
    ['LDL', 'cholesterol_ldl'],
    ['Hemoglobin', 'hemoglobin'],
    ['Temperature', 'temperature'],
    ['O₂ Saturation', 'oxygen_saturation'],
  ]

  return (
    <div className="fixed inset-0 bg-black/40 backdrop-blur-sm z-50 flex items-center justify-center p-4 overflow-y-auto">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-lg my-4">
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100 sticky top-0 bg-white rounded-t-2xl">
          <h2 className="text-base font-semibold text-slate-800">Add Health Record</h2>
          <button onClick={onClose}><X className="w-5 h-5 text-slate-400" /></button>
        </div>
        <div className="p-6 grid grid-cols-2 gap-3">
          {fields.map(([label, key]) => (
            <div key={key}>
              <label className="label">{label}</label>
              <input className="input" type="number" step="0.1"
                value={form[key] as number}
                onChange={e => setForm(f => ({ ...f, [key]: +e.target.value }))}
              />
            </div>
          ))}
          <div className="col-span-2">
            <label className="label">Notes</label>
            <textarea className="input h-16 resize-none" value={form.notes}
              onChange={e => setForm(f => ({ ...f, notes: e.target.value }))} />
          </div>
        </div>
        <div className="flex gap-3 px-6 pb-5">
          <button className="btn-secondary flex-1" onClick={onClose}>Cancel</button>
          <button className="btn-primary flex-1" onClick={() => mutation.mutate()} disabled={mutation.isPending}>
            {mutation.isPending ? 'Saving…' : 'Save Record'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default function PatientDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const qc = useQueryClient()
  const pid = Number(id)
  const [showRecord, setShowRecord] = useState(false)
  const [predicting, setPredicting] = useState(false)

  const { data: patient, isLoading: pLoading } = useQuery<Patient>({
    queryKey: ['patient', pid],
    queryFn: async () => { const r = await patientApi.get(pid); return r.data },
  })

  const { data: records = [] } = useQuery<HealthRecord[]>({
    queryKey: ['records', pid],
    queryFn: async () => { const r = await patientApi.getRecords(pid); return r.data },
  })

  const { data: history = [] } = useQuery<any[]>({
    queryKey: ['pred-history', pid],
    queryFn: async () => { const r = await predictApi.history(pid); return r.data },
  })

  const predictMutation = useMutation({
    mutationFn: (recordId?: number) => predictApi.predict({
      patient_id: pid,
      ...(recordId ? { health_record_id: recordId } : {}),
    }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['pred-history', pid] }),
    onSettled: () => setPredicting(false),
  })

  const trendData = [...history].reverse().map((p, i) => ({
    i: i + 1, score: Math.round(p.risk_score * 100), level: p.risk_level,
  }))

  const latestPred = history[0]

  if (pLoading) return <Spinner />
  if (!patient) return <p className="text-slate-400 text-sm">Patient not found</p>

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <button onClick={() => navigate(-1)} className="btn-secondary p-2">
          <ArrowLeft className="w-4 h-4" />
        </button>
        <div>
          <h1 className="text-xl font-bold text-slate-800">{patient.patient_code}</h1>
          <p className="text-sm text-slate-400 capitalize">{patient.age} yrs · {patient.gender} · {patient.blood_group || 'Unknown BG'}</p>
        </div>
      </div>

      {/* Top row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Patient info */}
        <div className="card p-5 space-y-3">
          <h2 className="text-sm font-semibold text-slate-700 mb-2">Patient Info</h2>
          {[
            ['Age', `${patient.age} years`],
            ['Gender', patient.gender],
            ['Blood Group', patient.blood_group || 'Not recorded'],
            ['Contact', patient.contact_number || 'N/A'],
          ].map(([k, v]) => (
            <div key={k} className="flex justify-between text-sm">
              <span className="text-slate-400">{k}</span>
              <span className="text-slate-700 font-medium capitalize">{v}</span>
            </div>
          ))}
        </div>

        {/* Latest risk score */}
        <div className="card p-5 flex flex-col items-center justify-center gap-3">
          {latestPred ? (
            <>
              <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">Latest Risk Score</p>
              <RiskScoreRing score={latestPred.risk_score} level={latestPred.risk_level} />
              <RiskBadge level={latestPred.risk_level} />
              <p className="text-xs text-slate-400">{formatDate(latestPred.created_at)}</p>
            </>
          ) : (
            <div className="text-center">
              <Brain className="w-8 h-8 text-slate-300 mx-auto mb-2" />
              <p className="text-sm text-slate-400">No prediction yet</p>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="card p-5 space-y-3">
          <h2 className="text-sm font-semibold text-slate-700 mb-2">Actions</h2>
          <button
            className="btn-primary w-full flex items-center justify-center gap-2"
            onClick={() => { setPredicting(true); predictMutation.mutate(records[0]?.id) }}
            disabled={predictMutation.isPending}
          >
            <Brain className="w-4 h-4" />
            {predictMutation.isPending ? 'Predicting…' : 'Run Risk Prediction'}
          </button>
          <button
            className="btn-secondary w-full flex items-center justify-center gap-2"
            onClick={() => setShowRecord(true)}
          >
            <Plus className="w-4 h-4" /> Add Health Record
          </button>
          <button
            className="btn-secondary w-full flex items-center justify-center gap-2"
            onClick={() => navigate('/triage')}
          >
            <Activity className="w-4 h-4" /> AI Symptom Triage
          </button>
        </div>
      </div>

      {/* Risk trend chart */}
      {trendData.length > 1 && (
        <SectionCard title="Risk Score History">
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={trendData} margin={{ top: 5, right: 10, bottom: 0, left: -10 }}>
              <XAxis dataKey="i" tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} axisLine={false} label={{ value: 'Prediction #', position: 'insideBottom', offset: -2, fontSize: 11, fill: '#94a3b8' }} />
              <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} axisLine={false} domain={[0, 100]} unit="%" />
              <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} formatter={(v: number) => [`${v}%`, 'Risk']} />
              <ReferenceLine y={75} stroke="#ef4444" strokeDasharray="3 3" label={{ value: 'High', fontSize: 10, fill: '#ef4444' }} />
              <ReferenceLine y={50} stroke="#f97316" strokeDasharray="3 3" />
              <Line type="monotone" dataKey="score" stroke="#6366f1" strokeWidth={2} dot={{ r: 3, fill: '#6366f1' }} />
            </LineChart>
          </ResponsiveContainer>
        </SectionCard>
      )}

      {/* Health records */}
      <SectionCard
        title={`Health Records (${records.length})`}
        action={
          <button className="btn-secondary text-xs py-1.5" onClick={() => setShowRecord(true)}>
            <Plus className="w-3 h-3 inline mr-1" />Add
          </button>
        }
      >
        {records.length === 0 ? (
          <p className="text-sm text-slate-400 text-center py-4">No records yet</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-slate-400 uppercase tracking-wide border-b border-slate-100">
                  {['Date', 'BP', 'HR', 'Glucose', 'BMI', 'O₂%'].map(h => (
                    <th key={h} className="text-left pb-2 font-medium pr-4">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-50">
                {records.map(r => (
                  <tr key={r.id} className="text-slate-600">
                    <td className="py-2 pr-4 text-slate-400 text-xs">{formatDate(r.recorded_at)}</td>
                    <td className="py-2 pr-4">{r.blood_pressure_systolic}/{r.blood_pressure_diastolic}</td>
                    <td className="py-2 pr-4">{r.heart_rate}</td>
                    <td className="py-2 pr-4">{r.blood_glucose}</td>
                    <td className="py-2 pr-4">{r.bmi}</td>
                    <td className="py-2 pr-4">{r.oxygen_saturation}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </SectionCard>

      {/* Prediction history */}
      {history.length > 0 && (
        <SectionCard title="Prediction History">
          <div className="space-y-3">
            {history.map((p: any) => (
              <div key={p.id} className="flex items-start gap-4 p-3 bg-slate-50 rounded-xl">
                <div className="text-center min-w-[52px]">
                  <p className="text-lg font-bold text-slate-800">{Math.round(p.risk_score * 100)}%</p>
                  <RiskBadge level={p.risk_level} />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-slate-400 mb-1">{formatDateTime(p.created_at)} · {p.model_version}</p>
                  {p.top_risk_factors?.slice(0, 3).map((f: any, i: number) => (
                    <span key={i} className="inline-block text-xs bg-white border border-slate-200 rounded-full px-2 py-0.5 mr-1 mb-1 text-slate-600">
                      {f.feature}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </SectionCard>
      )}

      {showRecord && <AddRecordModal patientId={pid} onClose={() => setShowRecord(false)} />}
    </div>
  )
}
