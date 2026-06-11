import { X } from "lucide-react"
import { HistoryItem } from "./HistoryItem"
import type { DocumentHistory } from "../types/document.types"

interface RecentDocumentsProps {
    documents: DocumentHistory[]
    active_doc_id: string
    onSelectDocument: (doc_id: string, session_id: string) => void
    onRemoveDocument: (doc_id: string) => void
    onClose: () => void
    isOpen: boolean
}

export const RecentDocuments = ({ documents, active_doc_id, onSelectDocument, onRemoveDocument, onClose, isOpen }: RecentDocumentsProps) => {
    return (
        <>
            {/* Overlay oscuro */}
            {isOpen && (
                <div
                    className="fixed inset-0 bg-black/40 z-51 backdrop-blur-[2px] transition-opacity"
                    onClick={onClose}
                />
            )}

            {/* Panel lateral */}
            <section
                className={`fixed top-0 right-0 h-full z-52 w-72 shrink-0 bg-surface border-l border-outline-variant flex flex-col transition-transform duration-300 ease-in-out ${isOpen ? "translate-x-0 shadow-2xl" : "translate-x-full"
                    }`}
            >
                <div className="p-4 border-b border-outline-variant flex items-center justify-between">
                    <h2 className="font-geist text-[14px] font-bold uppercase tracking-widest text-on-surface-variant">
                        Recientes
                    </h2>
                    <button
                        onClick={onClose}
                        className="p-1 rounded-md text-on-surface-variant hover:text-on-surface hover:bg-surface-container transition-colors cursor-pointer"
                    >
                        <X size={18} />
                    </button>
                </div>

                {/* Lista de documentos scrolleable */}
                <div className="flex-1 overflow-y-auto custom-scrollbar p-3 chat-scrollbar flex flex-col gap-2">
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
                            No hay documentos recientes.
                        </p>
                    )}
                </div>
            </section>
        </>
    )
}
