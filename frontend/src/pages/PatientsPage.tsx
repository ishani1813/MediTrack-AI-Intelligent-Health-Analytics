import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { Plus, Search, User, ChevronRight, X } from 'lucide-react'
import { patientApi } from '../services/api'
import type { Patient, PatientCreate } from '../types'
import { PageHeader, Spinner, EmptyState } from '../components/shared'
import { formatDate } from '../utils/helpers'

function CreatePatientModal({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient()
  const [form, setForm] = useState<PatientCreate>({
    age: 35, gender: 'male', blood_group: '', contact_number: '', address: '', medical_history: {},
  })
  const [error, setError] = useState('')

  const mutation = useMutation({
    mutationFn: () => patientApi.create(form),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['patients'] }); onClose() },
    onError: (e: any) => setError(e.response?.data?.detail || 'Failed to create patient'),
  })

  const field = (label: string, key: keyof PatientCreate, type = 'text', opts?: any) => (
    <div>
      <label className="label">{label}</label>
      {opts?.options ? (
        <select className="input" value={form[key] as string}
          onChange={e => setForm(f => ({ ...f, [key]: e.target.value }))}>
          {opts.options.map((o: string) => <option key={o} value={o}>{o}</option>)}
        </select>
      ) : (
        <input className="input" type={type}
          value={form[key] as string | number}
          onChange={e => setForm(f => ({ ...f, [key]: type === 'number' ? +e.target.value : e.target.value }))}
        />
      )}
    </div>
  )

  return (
    <div className="fixed inset-0 bg-black/40 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-md">
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100">
          <h2 className="text-base font-semibold text-slate-800">New Patient</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600"><X className="w-5 h-5" /></button>
        </div>
        <div className="p-6 space-y-4">
          <div className="grid grid-cols-2 gap-4">
            {field('Age', 'age', 'number')}
            {field('Gender', 'gender', 'text', { options: ['male', 'female', 'other'] })}
          </div>
          <div className="grid grid-cols-2 gap-4">
            {field('Blood Group', 'blood_group', 'text')}
            {field('Contact Number', 'contact_number', 'tel')}
          </div>
          {field('Address', 'address')}
          {error && <p className="text-xs text-red-500 bg-red-50 px-3 py-2 rounded-lg">{error}</p>}
        </div>
        <div className="flex gap-3 px-6 pb-5">
          <button className="btn-secondary flex-1" onClick={onClose}>Cancel</button>
          <button className="btn-primary flex-1" onClick={() => mutation.mutate()} disabled={mutation.isPending}>
            {mutation.isPending ? 'Creating…' : 'Create Patient'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default function PatientsPage() {
  const navigate = useNavigate()
  const [search, setSearch] = useState('')
  const [showCreate, setShowCreate] = useState(false)

  const { data: patients = [], isLoading } = useQuery<Patient[]>({
    queryKey: ['patients'],
    queryFn: async () => { const res = await patientApi.list(); return res.data },
  })

  const filtered = patients.filter(p =>
    p.patient_code.toLowerCase().includes(search.toLowerCase()) ||
    String(p.age).includes(search) ||
    p.gender.includes(search.toLowerCase())
  )

  return (
    <div>
      <PageHeader
        title="Patients"
        subtitle={`${patients.length} patient records`}
        action={
          <button className="btn-primary flex items-center gap-2" onClick={() => setShowCreate(true)}>
            <Plus className="w-4 h-4" /> New Patient
          </button>
        }
      />

      {/* Search */}
      <div className="relative mb-4">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
        <input
          className="input pl-9"
          placeholder="Search by code, age, or gender…"
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
      </div>

      {isLoading ? (
        <Spinner label="Loading patients…" />
      ) : filtered.length === 0 ? (
        <EmptyState
          icon={<User className="w-6 h-6" />}
          title="No patients found"
          description={search ? 'Try a different search term' : 'Create your first patient record'}
          action={!search && (
            <button className="btn-primary" onClick={() => setShowCreate(true)}>
              <Plus className="w-4 h-4 inline mr-1" />Add Patient
            </button>
          )}
        />
      ) : (
        <div className="card overflow-hidden">
          {/* Table header */}
          <div className="hidden md:grid grid-cols-5 gap-4 px-5 py-3 bg-slate-50 border-b border-slate-100 text-xs font-medium text-slate-500 uppercase tracking-wide">
            <span>Code</span><span>Age</span><span>Gender</span><span>Blood Group</span><span>Created</span>
          </div>
          {/* Rows */}
          <div className="divide-y divide-slate-100">
            {filtered.map(p => (
              <button
                key={p.id}
                onClick={() => navigate(`/patients/${p.id}`)}
                className="w-full grid grid-cols-2 md:grid-cols-5 gap-4 px-5 py-3.5 text-sm text-left hover:bg-slate-50 transition-colors items-center"
              >
                <span className="font-medium text-indigo-600">{p.patient_code}</span>
                <span className="text-slate-600">{p.age} yrs</span>
                <span className="hidden md:block capitalize text-slate-600">{p.gender}</span>
                <span className="hidden md:block text-slate-600">{p.blood_group || '—'}</span>
                <div className="hidden md:flex items-center justify-between">
                  <span className="text-slate-400">{formatDate(p.created_at)}</span>
                  <ChevronRight className="w-4 h-4 text-slate-300" />
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {showCreate && <CreatePatientModal onClose={() => setShowCreate(false)} />}
    </div>
  )
}
