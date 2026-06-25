import { Component } from "react"
import type { ErrorInfo, ReactNode } from "react"
import { AlertTriangle } from "lucide-react"

interface ErrorBoundaryProps {
    children: ReactNode
}

interface ErrorBoundaryState {
    hasError: boolean
}

const FALLBACK_TITLE = "Algo ha fallado"
const FALLBACK_DESCRIPTION = "Ha ocurrido un error inesperado en la aplicación. Puedes intentar recargar la página."
const RELOAD_BUTTON_LABEL = "Recargar página"

// Excepción deliberada al patrón funcional del proyecto: los Error Boundaries
// de React solo pueden implementarse con componentes de clase (getDerivedStateFromError
// y componentDidCatch no tienen equivalente en hooks a fecha de React 19).
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
    state: ErrorBoundaryState = { hasError: false }

    static getDerivedStateFromError(): ErrorBoundaryState {
        return { hasError: true }
    }

    componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
        console.error("[ErrorBoundary] Error no controlado en el árbol de renderizado:", error, errorInfo)
    }

    handleReload = (): void => {
        window.location.reload()
    }

    render(): ReactNode {
        if (!this.state.hasError) {
            return this.props.children
        }

        return (
            <div
                role="alert"
                data-testid="error-boundary-fallback"
                className="flex flex-col items-center justify-center h-screen bg-surface p-6 gap-4 text-center"
            >
                <AlertTriangle className="text-error" size={40} aria-hidden="true" />
                <h1 className="font-geist text-headline-lg text-on-surface">{FALLBACK_TITLE}</h1>
                <p className="font-geist text-body-md text-on-surface-variant max-w-md">
                    {FALLBACK_DESCRIPTION}
                </p>
                <button
                    type="button"
                    data-testid="error-boundary-reload"
                    onClick={this.handleReload}
                    className="px-4 py-2 rounded-lg bg-primary text-on-primary font-jetbrains text-label-md font-bold hover:opacity-90 active:scale-95 transition-all cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-surface"
                >
                    {RELOAD_BUTTON_LABEL}
                </button>
            </div>
        )
    }
}
