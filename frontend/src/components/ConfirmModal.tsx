interface ConfirmModalProps {
    onConfirm: () => void;
    onCancel: () => void;
}

export const ConfirmModal = ({ onConfirm, onCancel }: ConfirmModalProps) => {
    return (
        <div className="fixed inset-0 z-51 flex items-center justify-center p-4 modal-backdrop backdrop-blur-sm bg-black/6">
            <div className="bg-surface-container-high rounded-lg border border-message-ai shadow-xl max-w-md w-full p-6 flex flex-col gap-6 animate-in zoom-in-95 duration-200">
                <div className="flex flex-col gap-2">
                    <h3 className="text-headline-md font-geist font-bold text-on-surface">¿Eliminar este documento?</h3>
                    <p className="font-geist text-body-md text-on-surface-variant leading-relaxed">
                        Esta acción eliminará el documento y todo su historial de conversación. No se puede deshacer. ¿Quieres continuar?
                    </p>
                </div>
                <div className="flex items-center justify-end gap-3">
                    <button
                        onClick={onCancel}
                        className="px-4 py-2 rounded-lg font-jetbrains text-label-md font-medium text-on-surface-variant hover:bg-surface-variant transition-colors cursor-pointer"
                        id="cancel-upload"
                    >
                        Cancelar
                    </button>
                    <button
                        onClick={onConfirm}
                        className="px-4 py-2 rounded-lg bg-error text-on-error font-jetbrains text-label-md font-bold hover:opacity-90 active:scale-95 transition-all cursor-pointer"
                        id="confirm-upload"
                    >
                        Confirmar
                    </button>
                </div>
            </div>
        </div>
    )
}
