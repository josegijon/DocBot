import { useFocusTrap } from "../hooks/useFocusTrap"

interface ConfirmModalProps {
    title: string
    description: string
    confirmText?: string
    cancelText?: string
    onConfirm: () => void
    onCancel: () => void
}

const MODAL_TITLE_ID = "confirm-modal-title"
const MODAL_DESCRIPTION_ID = "confirm-modal-description"

export const ConfirmModal = ({
    title,
    description,
    confirmText = "Confirmar",
    cancelText = "Cancelar",
    onConfirm,
    onCancel,
}: ConfirmModalProps) => {
    const { containerRef } = useFocusTrap<HTMLDivElement>({ isOpen: true, onClose: onCancel })

    return (
        <>
            <div
                className="fixed inset-0 z-51 modal-backdrop backdrop-blur-sm bg-black/6 cursor-default"
                aria-hidden="true"
                onClick={onCancel}
            />

            <div className="fixed inset-0 z-51 flex items-center justify-center p-4 pointer-events-none">
                <div
                    ref={containerRef}
                    role="dialog"
                    aria-modal="true"
                    aria-labelledby={MODAL_TITLE_ID}
                    aria-describedby={MODAL_DESCRIPTION_ID}
                    tabIndex={-1}
                    className="pointer-events-auto bg-surface-container-high rounded-lg border border-message-ai shadow-xl max-w-md w-full p-6 flex flex-col gap-6 animate-in zoom-in-95 duration-200"
                >
                    <div className="flex flex-col gap-2">
                        <h3 id={MODAL_TITLE_ID} className="text-headline-md font-geist font-bold text-on-surface">
                            {title}
                        </h3>
                        <p id={MODAL_DESCRIPTION_ID} className="font-geist text-body-md text-on-surface-variant leading-relaxed">
                            {description}
                        </p>
                    </div>
                    <div className="flex items-center justify-end gap-3">
                        <button
                            data-testid="confirm-modal-cancel"
                            onClick={onCancel}
                            className="px-4 py-2 rounded-lg font-jetbrains text-label-md font-medium text-on-surface-variant hover:bg-surface-variant transition-colors cursor-pointer"
                        >
                            {cancelText}
                        </button>
                        <button
                            data-testid="confirm-modal-confirm"
                            onClick={onConfirm}
                            className="px-4 py-2 rounded-lg bg-error text-on-error font-jetbrains text-label-md font-bold hover:opacity-90 active:scale-95 transition-all cursor-pointer"
                        >
                            {confirmText}
                        </button>
                    </div>
                </div>
            </div>
        </>
    )
}
