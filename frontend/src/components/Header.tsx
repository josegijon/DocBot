import docbotLogo from "../assets/img/docbot-sin-bg.png"


export const Header = () => {
    return (
        <header className="fixed top-0 w-full bg-surface border-b border-outline-variant flex items-center justify-between p-3">
            <div className="mx-auto w-full max-w-480">
                <h1 className="font-geist text-[25px] font-semibold text-on-surface tracking-[-0.04em] flex items-center justify-center">
                    <img
                        src={docbotLogo}
                        alt="DocBot Logo"
                        className="h-10 w-10 object-contain mr-1"
                    />
                    Doc<span className="font-bold text-primary">Bot</span>
                </h1>
            </div>
        </header>
    )
}
