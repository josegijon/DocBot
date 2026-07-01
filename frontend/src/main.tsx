import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Toaster } from 'sonner'

import App from './App.tsx'
import './index.css'
import { ErrorBoundary } from './components/ErrorBoundary.tsx'
import { Analytics } from '@vercel/analytics/react'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Toaster />
    <Analytics />
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
