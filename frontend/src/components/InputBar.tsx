import { Loader2, Send } from "lucide-react"
import { useState, useRef, useEffect } from "react"

const KEY_ENTER = "Enter"
const INPUT_TEXTAREA_ID = "chat-input-textarea"
const TEXTAREA_LABEL = "Escribe tu pregunta"
const SEND_BUTTON_LABEL = "Enviar mensaje"
const LOADING_ANNOUNCEMENT = "DocBot está generando una respuesta. Espera un momento."

interface InputBarProps {
    onSend: (message: string) => void
    isLoading: boolean
    disabled: boolean
}

export const InputBar = ({ onSend, isLoading, disabled }: InputBarProps) => {
    const [input, setInput] = useState<string>("")
    const textareaRef = useRef<HTMLTextAreaElement>(null)
    const formRef = useRef<HTMLFormElement>(null)

    const hadFocusWithinBarRef = useRef(false)

    const handleSend = () => {
        if (!input.trim() || isLoading) return
        hadFocusWithinBarRef.current = !!formRef.current?.contains(document.activeElement)
        onSend(input.trim())
        setInput("")
    }

    const handleSubmit = (e: React.SubmitEvent<HTMLFormElement>) => {
        e.preventDefault()
        handleSend()
    }

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key !== KEY_ENTER || e.shiftKey) return
        e.preventDefault()
        handleSend()
    }

    useEffect(() => {
        if (!disabled) {
            textareaRef.current?.focus()
        }
    }, [disabled])

    useEffect(() => {
        if (isLoading || !hadFocusWithinBarRef.current) return

        if (document.activeElement === document.body) {
            textareaRef.current?.focus()
        }
        hadFocusWithinBarRef.current = false
    }, [isLoading])

    return (
        <div className="p-6 bg-surface">
            <form
                ref={formRef}
                className="max-w-4xl mx-auto relative"
                onSubmit={handleSubmit}
                aria-busy={isLoading}
            >
                <div className="border border-[#333] rounded-xl bg-surface-container-low shadow-lg focus-within:ring-1 focus-within:ring-primary/50 transition-all overflow-hidden flex items-center">
                    <label htmlFor={INPUT_TEXTAREA_ID} className="sr-only">
                        {TEXTAREA_LABEL}
                    </label>
                    <textarea
                        id={INPUT_TEXTAREA_ID}
                        ref={textareaRef}
                        rows={1}
                        className="w-full bg-transparent border-none focus:outline-none focus:ring-0 font-body-md text-body-md text-on-surface p-4 resize-none min-h-14 chat-scrollbar"
                        placeholder={disabled ? "Sube un PDF para empezar..." : "Escribe tu pregunta..."}
                        disabled={isLoading || disabled}
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                    >
                    </textarea>
                    <div className="flex items-center justify-center px-4">
                        <div className="flex items-center justify-center gap-2">
                            <button
                                type="submit"
                                aria-label={SEND_BUTTON_LABEL}
                                title={SEND_BUTTON_LABEL}
                                className="p-2 text-on-primary bg-primary rounded-lg transition-all cursor-pointer flex items-center justify-center hover:opacity-90 active:scale-95 disabled:bg-surface-variant disabled:text-outline disabled:cursor-default focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                                disabled={isLoading || disabled || !input.trim()}
                            >
                                {isLoading
                                    ? <Loader2 size={18} className="animate-spin" aria-hidden="true" />
                                    : <Send size={18} aria-hidden="true" />}
                            </button>
                        </div>
                    </div>
                </div>
                <div aria-live="polite" aria-atomic="true" className="sr-only">
                    {isLoading ? LOADING_ANNOUNCEMENT : ""}
                </div>
            </form>
            <p className="text-center mt-3 font-jetbrains text-label-md text-on-surface-variant opacity-85">
                DocBot puede cometer errores. Recuerda verificar la información importante.
            </p>
        </div>
    )
}
