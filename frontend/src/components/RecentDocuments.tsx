import { useEffect, useRef } from "react"
import { X } from "lucide-react"
import { HistoryItem } from "./HistoryItem"
import type { DocumentHistory } from "../types/document.types"

interface RecentDocumentsProps {
    documents: DocumentHistory[]
    active_doc_id: string | null
    onSelectDocument: (doc_id: string, session_id: string) => void
    onRemoveDocument: (doc_id: string) => void
    onClose: () => void
    isOpen: boolean
}

const PANEL_TITLE = "Recientes"
const EMPTY_STATE_TEXT = "No hay documentos recientes."

export const RecentDocuments = ({ documents, active_doc_id, onSelectDocument, onRemoveDocument, onClose, isOpen }: RecentDocumentsProps) => {

    const sectionRef = useRef<HTMLElement>(null)
    const previousFocusRef = useRef<HTMLElement | null>(null)

    const sectionClass = isOpen
        ? "translate-x-0 shadow-2xl"
        : "translate-x-full"

    useEffect(() => {
        if (!isOpen) return

        previousFocusRef.current = document.activeElement as HTMLElement

        sectionRef.current?.focus()

        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") {
                e.preventDefault()
                onClose()
            }

            // Focus trap: cicla dentro del panel en Tab y Shift+Tab
            if (e.key === "Tab") {
                const focusable = sectionRef.current?.querySelectorAll<HTMLElement>(
                    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
                )
                if (!focusable || focusable.length === 0) return

                const first = focusable[0]
                const last = focusable[focusable.length - 1]

                if (e.shiftKey) {
                    // Shift+Tab desde el primer elemento → salta al último
                    if (document.activeElement === first) {
                        e.preventDefault()
                        last.focus()
                    }
                } else {
                    // Tab desde el último elemento → salta al primero
                    if (document.activeElement === last) {
                        e.preventDefault()
                        first.focus()
                    }
                }
            }
        }

        document.addEventListener("keydown", handleKeyDown)
        return () => {
            document.removeEventListener("keydown", handleKeyDown)
            previousFocusRef.current?.focus()
        }
    }, [isOpen, onClose])

    return (
        <>
            {isOpen && (
                <button
                    className="fixed inset-0 bg-black/40 z-51 backdrop-blur-[2px] transition-opacity"
                    onClick={onClose}
                    aria-label="Cerrar panel de documentos recientes"
                    tabIndex={-1}
                />
            )}

            <section
                ref={sectionRef}
                className={`fixed top-0 right-0 h-full z-52 w-72 shrink-0 bg-surface border-l border-outline-variant flex flex-col transition-transform duration-300 ease-in-out ${sectionClass}`}
                role="dialog"
                aria-modal="true"
                aria-labelledby="recent-docs-panel-title"
                tabIndex={-1}
            >
                <div className="p-4 border-b border-outline-variant flex items-center justify-between">
                    <h2
                        id="recent-docs-panel-title"
                        className="font-geist text-sm  font-bold uppercase tracking-widest text-on-surface-variant"
                    >
                        {PANEL_TITLE}
                    </h2>
                    <button
                        onClick={onClose}
                        title="Cerrar"
                        className="p-1 rounded-md text-on-surface-variant hover:text-on-surface hover:bg-surface-container transition-colors cursor-pointer"
                    >
                        <X size={18} aria-hidden="true" />
                    </button>
                </div>

                <div
                    className="flex-1 overflow-y-auto p-3 chat-scrollbar flex flex-col gap-2"
                    aria-live="polite"
                    aria-atomic="true"
                >
                    {documents.map((doc) => (
                        <HistoryItem
                            key={doc.doc_id}
                            doc_id={doc.doc_id}
                            session_id={doc.session_id}
                            filename={doc.filename}
                            saved_at={doc.saved_at}
                            isActive={doc.doc_id === active_doc_id}
                            onSelect={onSelectDocument}
                            onRemove={onRemoveDocument}
                        />
                    ))}

                    {documents.length === 0 && (
                        <p className="text-center text-on-surface-variant text-body-sm mt-4 font-geist">
                            {EMPTY_STATE_TEXT}
                        </p>
                    )}
                </div>
            </section>
        </>
    )
}
