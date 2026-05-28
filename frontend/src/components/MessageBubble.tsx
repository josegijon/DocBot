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

    const formatMessageContent = (text: string) => {
        const regex = /(\[(?:P[áa]g|P[áa]gina)\.?\s*\d+\])/gi;
        const parts = text.split(regex);

        return parts.map((part, index) => {
            const isPageTag = part.match(/^\[(?:P[áa]g|P[áa]gina)\.?\s*(\d+)\]$/i);

            if (isPageTag) {
                const pageNumber = isPageTag[1];

                return (
                    <sup
                        key={index}
                        className="font-jetbrains text-primary cursor-help font-medium select-none text-xs"
                        title={`Página ${pageNumber}`}
                    >
                        [{pageNumber}]
                    </sup>
                );
            }
            return part;
        });
    };

    return (
        <div className={`flex flex-col gap-2 max-w-[90%] animate-in fade-in slide-in-from-bottom-3 duration-500 ease-out ${isUser ? "justify-end ml-auto" : "justify-start"}`}>
            <div className={`flex flex-col gap-3 p-4 rounded-xl shadow-sm ${isUser ? "rounded-tr-none bg-message-user border-border-message-user" : "rounded-tl-none bg-message-ai border-border-message-ai"}`}>

                <p className={`font-geist whitespace-pre-wrap leading-relaxed ${isUser ? "text-white" : "text-on-surface"}`}>
                    {formatMessageContent(message.content)}
                </p>

                {!isUser && message.sources && message.sources.length > 0 && (
                    <div className="border-t border-outline-variant pt-3 flex flex-col gap-2 animate-in fade-in duration-700 ease-out">
                        <div className="flex items-center gap-2">
                            <BookOpenText color="#c0c1ff" size={16} />
                            <p className="font-jetbrains text-[11px] text-on-surface-variant uppercase tracking-widest">Fuentes</p>
                        </div>

                        {message.sources.map((source, index) => (
                            <div
                                key={index}
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