import type { ReactNode } from 'react'

const PAGE_REFERENCE_SPLITTER = /([\[\(](?:P[áa]g|P[áa]gina)\.?\s*\d+(?:-\d+)?[\)\]])/gi;

export const formatMessageContent = (text: string): ReactNode[] => {
    const parts = text.split(PAGE_REFERENCE_SPLITTER);

    return parts.map((part, index) => {
        const pageReferenceMatch = part.match(/^[\[\(](?:P[áa]g|P[áa]gina)\.?\s*(\d+(?:-\d+)?)[\]\)]$/i);

        if (pageReferenceMatch) {
            const pageRange = pageReferenceMatch[1];

            const isRange = pageRange.includes("-");
            const tooltipText = isRange ? `Páginas ${pageRange}` : `Página ${pageRange}`;

            return (
                <sup
                    key={`ref-${index}`}
                    className="font-jetbrains text-primary cursor-help font-medium select-none text-xs"
                    title={tooltipText}
                >
                    [{pageRange}]
                </sup>
            );
        }
        return part;
    });
};

export const formatDate = (isoString: string): string => {
    const date = new Date(isoString)
    if (isNaN(date.getTime())) return '-'
    const now = new Date()

    const isToday = date.getDate() === now.getDate() &&
        date.getMonth() === now.getMonth() &&
        date.getFullYear() === now.getFullYear()

    const yesterday = new Date(now)
    yesterday.setDate(now.getDate() - 1)

    const isYesterday = date.getDate() === yesterday.getDate() &&
        date.getMonth() === yesterday.getMonth() &&
        date.getFullYear() === yesterday.getFullYear()

    const time = date.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' })

    if (isToday) return `Hoy, ${time}`
    if (isYesterday) return `Ayer`

    return date.toLocaleDateString('es-ES', { month: 'short', day: 'numeric', year: 'numeric' })
}
