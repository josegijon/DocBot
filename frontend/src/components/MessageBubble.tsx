import { BookOpenText } from "lucide-react"
import { SourceCard } from "./SourceCard"

interface Source {
    page: number
    text: string
    score?: number
}

interface Message {
    role: "user" | "assistant"
    content: string
    sources?: Source[]
}

interface MessageBubbleProps {
    message: Message
}

export const MessageBubble = ({ message }: MessageBubbleProps) => {
    const isUser = message.role === "user"

    return (
        <div className={`flex flex-col gap-2 max-w-[90%] animate-in fade-in slide-in-from-bottom-2 duration-500 ${isUser ? "justify-end delay-150 ml-auto" : "justify-start delay-300"}`}>
            <div className={`flex flex-col gap-3 p-4 rounded-xl ${isUser ? "rounded-tr-none bg-message-user border-border-message-user" : "rounded-tl-none bg-message-ai border-border-message-ai"}`}>
                <p className={`font-geist ${isUser ? "text-white" : "text-on-surface"}`}>
                    {message.content}
                </p>

                {!isUser && message.sources && message.sources.length > 0 && (
                    <div className="border-t border-outline-variant pt-3 flex flex-col gap-2">
                        <div className="flex items-center gap-2">
                            <BookOpenText color="#c0c1ff" size={16} />
                            <p className="font-jetbrains text-[11px] text-on-surface-variant uppercase tracking-widest">Fuentes</p>
                        </div>

                        {message.sources.map((source, index) => (
                            <SourceCard key={index} source={source} index={index + 1} />
                        ))}
                    </div>
                )}
            </div>
        </div>
    )
}