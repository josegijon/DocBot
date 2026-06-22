import type { Source } from "../types/chat.types"


interface SourceCardProps {
    source: Source
    index: number
}

export const SourceCard = ({ source, index }: SourceCardProps) => {
    return (
        <div className="p-2 bg-surface-container-low border border-outline-variant rounded-lg flex items-start gap-3 hover:border-primary transition-all">
            <div className="flex items-center gap-2">
                <span className="font-jetbrains text-label-md text-primary">
                    {index}
                </span>
                <div className="flex flex-col">
                    <span className="font-geist text-[13px] font-medium text-on-surface">"{source.text}"</span>
                    <span className="font-jetbrains text-2xs text-on-surface-variant">Pág {source.page}</span>
                </div>

            </div>
        </div>
    )
}
