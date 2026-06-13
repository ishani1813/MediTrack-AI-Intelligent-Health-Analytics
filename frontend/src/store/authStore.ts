import { create } from 'zustand'

interface AuthState {
  token: string | null
  userId: number | null
  role: string | null
  fullName: string | null
  isAuthenticated: boolean
  setAuth: (token: string, userId: number, role: string, fullName: string) => void
  logout: () => void
}

export const useAuthStore = create<AuthState>((set) => ({
  token: localStorage.getItem('token'),
  userId: Number(localStorage.getItem('userId')) || null,
  role: localStorage.getItem('role'),
  fullName: localStorage.getItem('fullName'),
  isAuthenticated: !!localStorage.getItem('token'),

  setAuth: (token, userId, role, fullName) => {
    localStorage.setItem('token', token)
    localStorage.setItem('userId', String(userId))
    localStorage.setItem('role', role)
    localStorage.setItem('fullName', fullName)
    set({ token, userId, role, fullName, isAuthenticated: true })
  },

  logout: () => {
    localStorage.clear()
    set({ token: null, userId: null, role: null, fullName: null, isAuthenticated: false })
  },
}))
