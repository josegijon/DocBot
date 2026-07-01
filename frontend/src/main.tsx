import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Toaster } from 'sonner'
import { Analytics } from "@vercel/analytics/next"

import App from './App.tsx'
import './index.css'
import { ErrorBoundary } from './components/ErrorBoundary.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Toaster />
    <Analytics />
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
