import axios from 'axios'

const API_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: API_URL,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

// Attach JWT token to every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// Handle 401 globally
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

// ── Auth ─────────────────────────────────────────────────────────────────────
export const authApi = {
  login: (email: string, password: string) =>
    api.post('/auth/login', { email, password }),
  register: (email: string, full_name: string, password: string, role = 'doctor') =>
    api.post('/auth/register', { email, full_name, password, role }),
}

// ── Patients ─────────────────────────────────────────────────────────────────
export const patientApi = {
  list: (skip = 0, limit = 50) => api.get(`/patients/?skip=${skip}&limit=${limit}`),
  get: (id: number) => api.get(`/patients/${id}`),
  create: (data: any) => api.post('/patients/', data),
  update: (id: number, data: any) => api.put(`/patients/${id}`, data),
  addRecord: (patientId: number, data: any) => api.post(`/patients/${patientId}/records`, data),
  getRecords: (patientId: number) => api.get(`/patients/${patientId}/records`),
}

// ── Prediction ────────────────────────────────────────────────────────────────
export const predictApi = {
  predict: (data: any) => api.post('/predict/risk', data),
  history: (patientId: number) => api.get(`/predict/history/${patientId}`),
}

// ── Triage ────────────────────────────────────────────────────────────────────
export const triageApi = {
  analyze: (data: any) => api.post('/triage/symptom', data),
}

// ── Analytics ─────────────────────────────────────────────────────────────────
export const analyticsApi = {
  dashboard: () => api.get('/analytics/dashboard'),
  cohort: () => api.get('/analytics/cohort'),
}
