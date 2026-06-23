import { FileText, FileUp, Plus, Sparkles } from "lucide-react"

interface ChatOnboardingProps {
    filename: string | null
}

export const ChatOnboarding = ({ filename }: ChatOnboardingProps) => {
    const hasDocument = filename !== null

    const mainIcon = hasDocument ? <FileText size={48} /> : <FileUp size={48} />
    const badgeIcon = hasDocument ? <Sparkles size={18} /> : <Plus size={18} />

    const onboardingMessage = hasDocument ? (
        <>
            <h2 className="font-geist text-headline-lg text-on-surface tracking-tight">
                Listo para explorar <span className="text-primary font-jetbrains italic">{filename}</span>
            </h2>
            <p className="font-geist text-body-lg text-on-surface-variant max-w-lg balance">
                La IA ya ha analizado el documento. Haz tu primera pregunta para comenzar la conversación.
            </p>
        </>
    ) : (
        <>
            <h2 className="font-geist text-headline-lg text-on-surface tracking-tight">
                Comienza a chatear con tus documentos
            </h2>
            <p className="font-geist text-body-lg text-on-surface-variant max-w-lg balance">
                Sube un archivo PDF para que la IA pueda analizarlo y responder a todas tus preguntas sobre su contenido.
            </p>
        </>
    )

    return (
        <div className="flex-1 flex flex-col items-center justify-center relative h-full p-6">
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-150 h-150 bg-primary/5 rounded-full blur-[120px] pointer-events-none"></div>

            <div className="max-w-3xl w-full flex flex-col items-center gap-4 text-center z-10">

                <div className="relative group">
                    <div className="absolute inset-0 bg-primary/20 blur-xl rounded-full scale-75 group-hover:scale-100 transition-transform duration-700"></div>
                    <div className="relative w-24 h-24 bg-surface-container border border-outline-variant rounded-2xl flex items-center justify-center text-primary">
                        {mainIcon}

                        <div className="absolute -bottom-2 -right-2 w-10 h-10 bg-surface-container-highest border border-outline-variant rounded-full flex items-center justify-center shadow-lg">
                            {badgeIcon}
                        </div>
                    </div>
                </div>

                {onboardingMessage}
            </div>
        </div>
    )
}
