import { TriangleAlert } from "lucide-react"

interface ConfirmModalProps {
    onConfirm: () => void;
    onCancel: () => void;
}

export const ConfirmModal = ({ onConfirm, onCancel }: ConfirmModalProps) => {
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 modal-backdrop backdrop-blur-sm bg-black/6" id="upload-modal">
            <div className="bg-surface-container-high border border-outline-variant rounded-xl w-full max-w-md shadow-2xl overflow-hidden" id="modal-content">
                <div className="p-6 border-b border-outline-variant">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-error-container flex items-center justify-center text-error">
                            <TriangleAlert />
                        </div>
                        <h3 className="font-geist text-headline-md text-on-surface">¿Eliminar este documento?</h3>
                    </div>
                </div>
                <div className="p-6">
                    <p className="font-geist text-body-md text-on-surface-variant leading-relaxed">
                        Esta acción eliminará el documento y todo su historial de conversación. No se puede deshacer. ¿Quieres continuar?
                    </p>
                </div>
                <div className="p-6 bg-surface-container flex items-center justify-end gap-3">
                    <button
                        onClick={onCancel}
                        className="px-5 py-2.5 rounded-lg font-jetbrains text-label-md text-on-surface-variant hover:bg-surface-variant transition-colors border border-outline-variant cursor-pointer"
                        id="cancel-upload"
                    >
                        Cancelar
                    </button>
                    <button
                        onClick={onConfirm}
                        className="px-5 py-2.5 rounded-lg font-jetbrains text-label-md bg-primary text-on-primary hover:opacity-90 active:scale-95 transition-all cursor-pointer"
                        id="confirm-upload"
                    >
                        Confirmar
                    </button>
                </div>
            </div>
        </div>
    )
}
