// src/components/MessageBubble.tsx
import { BookOpenText } from "lucide-react"
import { SourceCard } from "./SourceCard"
import { formatMessageContent } from "../utils/format"
import type { Message } from "../types/chat.types"

interface MessageBubbleProps {
    message: Message
}

export const MessageBubble = ({ message }: MessageBubbleProps) => {
    const isUser = message.role === "user"
    const sources = message.sources
    const shouldRenderSources = !isUser && sources && sources.length > 0

    const bubbleAlignmentClass = isUser ? "justify-end ml-auto" : "justify-start"

    const bubbleSurfaceClass = isUser
        ? "rounded-tr-none bg-message-user border-border-message-user"
        : "rounded-tl-none bg-message-ai border-border-message-ai"

    const bubbleTextColorClass = isUser ? "text-white" : "text-on-surface"

    return (
        <div className={`flex flex-col gap-2 max-w-[90%] animate-in fade-in slide-in-from-bottom-3 duration-500 ease-out ${bubbleAlignmentClass}`}>
            <div className={`flex flex-col gap-3 p-4 rounded-xl shadow-sm ${bubbleSurfaceClass}`}>

                <p className={`font-geist whitespace-pre-wrap leading-relaxed ${bubbleTextColorClass}`}>
                    {formatMessageContent(message.content)}
                </p>

                {shouldRenderSources && (
                    <div className="border-t border-outline-variant pt-3 flex flex-col gap-2 animate-in fade-in duration-700 ease-out">
                        <div className="flex items-center gap-2">
                            <BookOpenText className="text-primary" size={16} />
                            <p className="font-jetbrains text-2xs text-on-surface-variant uppercase tracking-widest">Fuentes</p>
                        </div>

                        {sources.map((source, index) => (
                            <div
                                key={`${source.page}-${source.text}`}
                                className="animate-in fade-in slide-in-from-bottom-2 duration-500 fill-mode-both"
                                style={{ animationDelay: `${index * 150}ms` }}
                            >
                                <SourceCard source={source} index={index + 1} />
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    )
}
