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