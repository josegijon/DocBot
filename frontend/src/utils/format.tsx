export const formatMessageContent = (text: string) => {
    const regex = /([\[\(](?:P[áa]g|P[áa]gina)\.?\s*\d+(?:-\d+)?[\)\]])/gi;
    const parts = text.split(regex);

    return parts.map((part, index) => {
        const isPageTag = part.match(/^[\[\(](?:P[áa]g|P[áa]gina)\.?\s*(\d+(?:-\d+)?)[\]\)]$/i);

        if (isPageTag) {
            const pageRange = isPageTag[1];

            const isRange = pageRange.includes("-");
            const tooltipText = isRange ? `Páginas ${pageRange}` : `Página ${pageRange}`;

            return (
                <sup
                    key={index}
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
    const now = new Date()

    const isToday = date.getDate() === now.getDate() &&
        date.getMonth() === now.getMonth() &&
        date.getFullYear() === now.getFullYear()

    const yesterday = new Date(now)
    yesterday.setDate(now.getDate() - 1)

    const isYesterday = date.getDate() === yesterday.getDate() &&
        date.getMonth() === yesterday.getMonth() &&
        date.getFullYear() === yesterday.getFullYear()

    const time = date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })

    if (isToday) return `Today, ${time}`
    if (isYesterday) return `Yesterday`

    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}