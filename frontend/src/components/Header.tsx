import { History } from "lucide-react"
import docbotLogo from "../assets/img/docbot-sin-bg.png"

interface HeaderProps {
    onOpenHistory: () => void
    isHistoryOpen: boolean
}

export const Header = ({ onOpenHistory, isHistoryOpen }: HeaderProps) => {
    return (
        <header className="fixed top-0 w-full bg-surface border-b border-outline-variant z-30">
            <div className="w-full max-w-480 mx-auto flex items-center justify-between p-3">
                <div className="w-10"></div>

                <div className="mx-auto">
                    <h1 className="font-geist text-headline-brand font-semibold text-on-surface tracking-[-0.04em] flex items-center justify-center">
                        <img
                            src={docbotLogo}
                            alt=""
                            className="h-10 w-10 object-contain mr-1"
                        />
                        Doc<span className="font-bold text-primary">Bot</span>
                    </h1>
                </div>

                <button
                    onClick={onOpenHistory}
                    aria-label="Abrir historial de documentos"
                    aria-expanded={isHistoryOpen}
                    aria-controls="recent-documents-panel"
                    title="Historial de documentos"
                    className="w-10 h-10 flex items-center justify-center text-on-surface-variant hover:text-primary hover:bg-surface-container rounded-lg transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                >
                    <History size={20} aria-hidden="true" />
                </button>
            </div>
        </header>
    )
}
