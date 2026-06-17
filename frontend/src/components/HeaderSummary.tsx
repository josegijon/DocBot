import { FileText } from "lucide-react"
import { formatFileSize } from "../utils/format"

interface HeaderSummaryProps {
    filename: string
    fileSizeBytes: number
}

export const HeaderSummary = ({ filename, fileSizeBytes }: HeaderSummaryProps) => {
    return (
        <div className='border-b border-outline-variant pb-6'>
            <div className='flex items-center gap-4'>
                <div className='w-10 h-10 border border-border-message-ai flex items-center justify-center bg-surface-container-high rounded-lg text-primary'>
                    <FileText />
                </div>
                <div className='flex-1 overflow-hidden'>
                    <h2 className='font-geist text-headline-md text-on-surface truncate'>
                        {filename}
                    </h2>
                    <span className='font-jetbrains text-code-sm text-on-surface-variant opacity-60'>
                        PDF · {formatFileSize(fileSizeBytes)}
                    </span>
                </div>
            </div>
        </div>
    )
}
