import { Send } from "lucide-react"
import { useState } from "react"

interface InputBarProps {
    onSend: (message: string) => void
    isLoading: boolean
    disabled: boolean
}

export const InputBar = ({ onSend, isLoading, disabled }: InputBarProps) => {
    const [input, setInput] = useState<string>("")

    const handleSend = () => {
        if (!input.trim() || isLoading) return
        onSend(input.trim())
        setInput("")
    }

    return (
        <div className="p-6 bg-surface">
            <div className="max-w-4xl mx-auto relative">
                <div className="border border-[#333] rounded-xl bg-surface-container-low shadow-lg focus-within:ring-1 focus-within:ring-primary/50 transition-all overflow-hidden flex items-center">
                    <textarea
                        rows={1}
                        className="w-full bg-transparent border-none focus:outline-none focus:ring-0 font-body-md text-body-md text-on-surface p-4 resize-none min-h-14 chat-scrollbar"
                        placeholder={disabled ? "Sube un PDF para empezar..." : "Escribe tu pregunta..."}
                        disabled={isLoading || disabled}
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === "Enter" && !e.shiftKey) {
                                e.preventDefault()
                                handleSend()
                            }
                        }}
                    >
                    </textarea>
                    <div className="flex items-center justify-center px-4">
                        <div className="flex items-center justify-center gap-2">
                            <button
                                className="p-2 text-on-primary bg-primary rounded-lg transition-all cursor-pointer flex items-center justify-center hover:opacity-90 active:scale-95 disabled:bg-surface-variant disabled:text-outline disabled:cursor-default"
                                onClick={handleSend}
                                disabled={isLoading || disabled || !input.trim()}
                            >
                                {isLoading ? "..." : <Send size={18} />}
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            <p className="text-center mt-3 font-jetbrains text-label-md text-on-surface-variant opacity-50">
                DocBot puede cometer errores. Recuerda verificar la información importante.
            </p>
        </div>
    )
}