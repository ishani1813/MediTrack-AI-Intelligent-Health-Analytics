// ── Auth ─────────────────────────────────────────────
export interface User {
  id: number
  email: string
  full_name: string
  role: 'admin' | 'doctor' | 'patient'
  is_active: boolean
  created_at: string
}

// ── Patient ──────────────────────────────────────────
export interface Patient {
  id: number
  patient_code: string
  age: number
  gender: 'male' | 'female' | 'other'
  blood_group?: string
  contact_number?: string
  address?: string
  medical_history?: Record<string, any>
  created_at: string
}

export interface PatientCreate {
  age: number
  gender: string
  blood_group?: string
  contact_number?: string
  address?: string
  medical_history?: Record<string, any>
}

// ── Health Record ─────────────────────────────────────
export interface HealthRecord {
  id: number
  patient_id: number
  blood_pressure_systolic?: number
  blood_pressure_diastolic?: number
  heart_rate?: number
  blood_glucose?: number
  bmi?: number
  cholesterol_total?: number
  cholesterol_hdl?: number
  cholesterol_ldl?: number
  hemoglobin?: number
  temperature?: number
  oxygen_saturation?: number
  notes?: string
  recorded_at: string
}

// ── Prediction ────────────────────────────────────────
export type RiskLevel = 'low' | 'medium' | 'high' | 'critical'

export interface ShapFeature {
  feature: string
  value: number
  shap_value: number
  impact: 'increases_risk' | 'decreases_risk'
}

export interface Prediction {
  prediction_id: number
  patient_id: number
  risk_score: number
  risk_level: RiskLevel
  confidence: number
  top_risk_factors: ShapFeature[]
  shap_summary: {
    features: string[]
    base_value: number
    top_positive: ShapFeature[]
    top_negative: ShapFeature[]
  }
  model_version: string
  recommendation: string
  cached: boolean
}

// ── Triage ────────────────────────────────────────────
export type UrgencyLevel = 'routine' | 'soon' | 'urgent' | 'emergency'

export interface RetrievedDoc {
  condition: string
  relevance_score: number
  excerpt: string
}

export interface TriageResult {
  session_id: number
  urgency_level: UrgencyLevel
  ai_assessment: string
  possible_conditions: string[]
  recommended_actions: string[]
  retrieved_references: RetrievedDoc[]
  disclaimer: string
}

// ── Analytics ─────────────────────────────────────────
export interface DashboardStats {
  total_patients: number
  records_this_month: number
  high_risk_count: number
  avg_risk_score: number
  triage_sessions_today: number
}

export interface RiskDistribution {
  low: number
  medium: number
  high: number
  critical: number
}

export interface TrendPoint {
  date: string
  value: number
}

export interface DashboardData {
  stats: DashboardStats
  risk_distribution: RiskDistribution
  risk_trend: TrendPoint[]
  top_risk_factors_global: { feature: string; avg_shap: number }[]
  recent_predictions: Prediction[]
}
