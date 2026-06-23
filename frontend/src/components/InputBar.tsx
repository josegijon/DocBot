import { Loader2, Send } from "lucide-react"
import { useState, useRef, useEffect } from "react"

const KEY_ENTER = "Enter"
const INPUT_TEXTAREA_ID = "chat-input-textarea"
const TEXTAREA_LABEL = "Escribe tu pregunta"
const SEND_BUTTON_LABEL = "Enviar mensaje"

interface InputBarProps {
    onSend: (message: string) => void
    isLoading: boolean
    disabled: boolean
}

export const InputBar = ({ onSend, isLoading, disabled }: InputBarProps) => {
    const [input, setInput] = useState<string>("")
    const textareaRef = useRef<HTMLTextAreaElement>(null)

    const handleSend = () => {
        if (!input.trim() || isLoading) return
        onSend(input.trim())
        setInput("")
    }

    const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault()
        handleSend()
    }

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key !== KEY_ENTER || e.shiftKey) return
        e.preventDefault()
        handleSend()
    }

    useEffect(() => {
        if (!isLoading && !disabled) {
            textareaRef.current?.focus()
        }
    }, [isLoading, disabled])

    return (
        <div className="p-6 bg-surface">
            <form className="max-w-4xl mx-auto relative" onSubmit={handleSubmit}>
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
                                className="p-2 text-on-primary bg-primary rounded-lg transition-all cursor-pointer flex items-center justify-center hover:opacity-90 active:scale-95 disabled:bg-surface-variant disabled:text-outline disabled:cursor-default"
                                disabled={isLoading || disabled || !input.trim()}
                            >
                                {isLoading
                                    ? <Loader2 size={18} className="animate-spin" aria-hidden="true" />
                                    : <Send size={18} aria-hidden="true" />}
                            </button>
                        </div>
                    </div>
                </div>
            </form>
            <p className="text-center mt-3 font-jetbrains text-label-md text-on-surface-variant opacity-50">
                DocBot puede cometer errores. Recuerda verificar la información importante.
            </p>
        </div>
    )
}
