import docbotLogo from "../assets/img/docbot-sin-bg.png"
import { History } from "lucide-react"

interface HeaderProps {
    onOpenHistory: () => void
}

export const Header = ({ onOpenHistory }: HeaderProps) => {
    return (
        <header className="fixed top-0 w-full bg-surface border-b border-outline-variant flex items-center justify-between p-3 z-30">
            <div className="w-10"></div>

            <div className="mx-auto">
                <h1 className="font-geist text-[25px] font-semibold text-on-surface tracking-[-0.04em] flex items-center justify-center">
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
                className="w-10 h-10 flex items-center justify-center text-on-surface-variant hover:text-primary hover:bg-surface-container rounded-lg transition-colors cursor-pointer"
                title="Historial de documentos"
            >
                <History size={20} aria-hidden="true" />
            </button>
        </header>
    )
}
