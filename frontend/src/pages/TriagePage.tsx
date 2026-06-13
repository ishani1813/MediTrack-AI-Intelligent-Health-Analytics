import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Stethoscope, Send, AlertTriangle, CheckCircle, Clock, Zap, FileText } from 'lucide-react'
import { triageApi } from '../services/api'
import type { TriageResult, UrgencyLevel } from '../types'
import { UrgencyBadge, PageHeader, SectionCard } from '../components/shared'
import { urgencyColors } from '../utils/helpers'

const SAMPLE_SYMPTOMS = [
  'Severe chest pain radiating to left arm, shortness of breath, sweating',
  'Persistent headache, blurred vision, blood pressure very high',
  'Extreme thirst, frequent urination, fatigue, blurred vision',
  'Shortness of breath, leg swelling, unable to lie flat',
  'Sudden weakness in left arm, slurred speech, facial drooping',
]

const URGENCY_ICONS: Record<UrgencyLevel, React.ReactNode> = {
  emergency: <Zap className="w-5 h-5" />,
  urgent:    <AlertTriangle className="w-5 h-5" />,
  soon:      <Clock className="w-5 h-5" />,
  routine:   <CheckCircle className="w-5 h-5" />,
}

export default function TriagePage() {
  const [symptoms, setSymptoms] = useState('')
  const [patientAge, setPatientAge] = useState<string>('')
  const [patientGender, setPatientGender] = useState('')
  const [history, setHistory] = useState<string>('')
  const [result, setResult] = useState<TriageResult | null>(null)

  const mutation = useMutation({
    mutationFn: () => triageApi.analyze({
      symptoms,
      patient_age: patientAge ? +patientAge : undefined,
      patient_gender: patientGender || undefined,
      medical_history: history ? history.split(',').map(s => s.trim()).filter(Boolean) : [],
    }),
    onSuccess: (res) => setResult(res.data),
  })

  const urgencyLevel = result?.urgency_level
  const uc = urgencyLevel ? urgencyColors[urgencyLevel] : null

  return (
    <div className="space-y-6">
      <PageHeader
        title="AI Symptom Triage"
        subtitle="LangChain RAG — retrieves from medical knowledge base and generates clinical assessment"
      />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Input panel */}
        <div className="card p-5 space-y-4">
          <div className="flex items-center gap-2 mb-1">
            <div className="w-7 h-7 bg-teal-50 rounded-lg flex items-center justify-center">
              <Stethoscope className="w-4 h-4 text-teal-600" />
            </div>
            <h2 className="text-sm font-semibold text-slate-700">Symptom Input</h2>
          </div>

          <div>
            <label className="label">Describe symptoms *</label>
            <textarea
              className="input h-28 resize-none"
              placeholder="e.g. Chest pain, shortness of breath, dizziness since 2 hours…"
              value={symptoms}
              onChange={e => setSymptoms(e.target.value)}
            />
          </div>

          {/* Quick fill examples */}
          <div>
            <p className="text-xs text-slate-400 mb-2">Quick examples:</p>
            <div className="flex flex-wrap gap-1.5">
              {SAMPLE_SYMPTOMS.map((s, i) => (
                <button key={i} onClick={() => setSymptoms(s)}
                  className="text-xs px-2 py-1 bg-slate-100 hover:bg-indigo-50 hover:text-indigo-600 text-slate-500 rounded-lg transition-colors text-left">
                  {s.slice(0, 38)}…
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="label">Patient Age</label>
              <input className="input" type="number" placeholder="Optional"
                value={patientAge} onChange={e => setPatientAge(e.target.value)} />
            </div>
            <div>
              <label className="label">Gender</label>
              <select className="input" value={patientGender} onChange={e => setPatientGender(e.target.value)}>
                <option value="">Not specified</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
            </div>
          </div>

          <div>
            <label className="label">Medical History (comma-separated)</label>
            <input className="input" placeholder="e.g. hypertension, diabetes"
              value={history} onChange={e => setHistory(e.target.value)} />
          </div>

          <button
            className="btn-primary w-full flex items-center justify-center gap-2"
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending || !symptoms.trim()}
          >
            <Send className="w-4 h-4" />
            {mutation.isPending ? 'Analyzing with AI…' : 'Analyze Symptoms'}
          </button>

          {mutation.isError && (
            <p className="text-xs text-red-500 bg-red-50 px-3 py-2 rounded-lg">
              Triage failed — check backend/LLM connection
            </p>
          )}
        </div>

        {/* Results panel */}
        {result ? (
          <div className="space-y-4">
            {/* Urgency banner */}
            <div className={`card p-5 ${uc?.bg} border-0`}>
              <div className="flex items-center gap-3">
                <div className={`${uc?.text}`}>{URGENCY_ICONS[urgencyLevel!]}</div>
                <div>
                  <div className="flex items-center gap-2 mb-0.5">
                    <UrgencyBadge level={urgencyLevel!} />
                    <span className="text-xs text-slate-500">Session #{result.session_id}</span>
                  </div>
                  <p className={`text-sm font-medium ${uc?.text}`}>{result.ai_assessment}</p>
                </div>
              </div>
            </div>

            {/* Possible conditions */}
            <SectionCard title="Possible Conditions">
              <div className="flex flex-wrap gap-2">
                {result.possible_conditions.map((c, i) => (
                  <span key={i} className="px-3 py-1.5 bg-slate-100 text-slate-700 text-sm rounded-lg font-medium">
                    {c}
                  </span>
                ))}
              </div>
            </SectionCard>

            {/* Recommended actions */}
            <SectionCard title="Recommended Actions">
              <ol className="space-y-2">
                {result.recommended_actions.map((a, i) => (
                  <li key={i} className="flex items-start gap-3 text-sm text-slate-700">
                    <span className="w-5 h-5 rounded-full bg-indigo-50 text-indigo-600 text-xs flex items-center justify-center shrink-0 mt-0.5 font-medium">
                      {i + 1}
                    </span>
                    {a}
                  </li>
                ))}
              </ol>
            </SectionCard>

            {/* RAG retrieved references */}
            {result.retrieved_references.length > 0 && (
              <SectionCard title="Retrieved Medical References (RAG)">
                <div className="space-y-3">
                  {result.retrieved_references.map((r, i) => (
                    <div key={i} className="flex items-start gap-3 p-3 bg-slate-50 rounded-xl">
                      <FileText className="w-4 h-4 text-slate-400 mt-0.5 shrink-0" />
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 mb-0.5">
                          <p className="text-sm font-medium text-slate-700">{r.condition}</p>
                          <span className="text-xs text-slate-400 bg-white border border-slate-200 px-1.5 py-0.5 rounded">
                            {(r.relevance_score * 100).toFixed(0)}% match
                          </span>
                        </div>
                        <p className="text-xs text-slate-400 leading-relaxed">{r.excerpt}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </SectionCard>
            )}

            {/* Disclaimer */}
            <div className="flex items-start gap-2 text-xs text-slate-400 bg-slate-50 rounded-xl px-4 py-3">
              <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5 text-amber-400" />
              {result.disclaimer}
            </div>
          </div>
        ) : (
          <div className="card flex flex-col items-center justify-center py-20 text-center">
            <div className="w-16 h-16 bg-teal-50 rounded-2xl flex items-center justify-center mb-4">
              <Stethoscope className="w-8 h-8 text-teal-400" />
            </div>
            <p className="text-sm font-medium text-slate-600">Triage assessment appears here</p>
            <p className="text-xs text-slate-400 mt-1 max-w-xs">
              Describe symptoms and click Analyze — AI retrieves from a medical knowledge base and generates a structured assessment
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
