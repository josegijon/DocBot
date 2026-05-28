import { ArrowDown } from "lucide-react"

interface ScrollToBottomProps {
    onClick: () => void
}

export const ScrollToBottom = ({ onClick }: ScrollToBottomProps) => {
    return (
        <button
            className="absolute bottom-0 left-1/2 -translate-x-1/2 flex items-center justify-center w-10 h-10 rounded-full bg-surface-bright border border-outline-variant shadow-lg text-on-surface opacity-80 hover:opacity-100 hover:bg-primary hover:text-on-primary transition-all duration-200 z-10 cursor-pointer"
            id="scroll-to-bottom-btn"
            onClick={onClick}
        >
            <ArrowDown />
        </button>
    )
}
