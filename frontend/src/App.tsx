import { useEffect, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { UploadZone } from './components/UploadZone';
import { useIngestionStatus } from './hooks/useIngestionStatus';
import { IngestionProgress } from './components/IngestionProgress';
import { DocumentSummary } from './components/DocumentSummary';
import { useSummary } from './hooks/useSummary';
import { Header } from './components/Header';
import { HeaderSummary } from './components/HeaderSummary';
import { ButtonNewDocument } from './components/ButtonNewDocument';
import { useChat } from './hooks/useChat';
import { ChatWindow } from './components/ChatWindow';
import { useDocumentHistory } from './hooks/useDocumentHistory';
import { RecentDocuments } from './components/RecentDocuments';
import { ConfirmModal } from './components/ConfirmModal';
import { FileText, MessagesSquare } from 'lucide-react';

export const App = () => {
  const [docId, setDocId] = useState<string | null>(null)
  const [filename, setFilename] = useState<string | null>(null)
  const [fileSize, setFileSize] = useState<number>(0)
  const [docToDelete, setDocToDelete] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string>(() => uuidv4())
  const [isHistoryOpen, setIsHistoryOpen] = useState(false)
  const [isDocumentReady, setIsDocumentReady] = useState<boolean>(false)
  const [activeTab, setActiveTab] = useState<"document" | "chat">("document")

  const { documents, addDocument, removeDocument } = useDocumentHistory()
  const { status, progress, resetStatus } = useIngestionStatus(docId, isDocumentReady)
  const { summary, isDone, resetSummary } = useSummary(docId, status)
  const { messages, isLoading, sendMessage, resetMessages } = useChat(docId, sessionId)

  // console.log(docId)
  // console.log(status, progress)
  // console.log(messages)
  // console.log(summary)

  const handleUploadSuccess = (docId: string, filename: string, fileSize: number) => {
    setDocId(docId)
    setFilename(filename)
    setFileSize(fileSize)
    addDocument(docId, sessionId, filename)
    setActiveTab('chat')
  }

  const handleNewDocument = () => {
    setIsDocumentReady(false)
    resetStatus()
    setDocId(null)
    setFilename(null)
    setFileSize(0)
    setSessionId(uuidv4())
    resetSummary()
  }

  const handleRemoveDocument = async (doc_id: string) => {
    const doc = documents.find(d => d.doc_id === doc_id)
    if (doc) localStorage.removeItem(`docbot_chat_${doc.session_id}`)

    if (doc_id === docId) handleNewDocument();
    await fetch(`/api/documents/${doc_id}`, { method: "DELETE" })
    removeDocument(doc_id)
  }

  useEffect(() => {
    if (status === "ready" && docId && filename) {
      const alreadyExists = documents.some(d => d.doc_id === docId)
      if (!alreadyExists) addDocument(docId, sessionId, filename)
    }
  }, [status])

  const handleSelectDocument = async (selectedDocId: string, selectedSessionId: string) => {
    if (selectedDocId === docId) {
      setIsHistoryOpen(false)
      return
    }

    const doc = documents.find(d => d.doc_id === selectedDocId)
    if (!doc) return

    const response = await fetch(`/api/documents/${selectedDocId}/exists`)
    const data = await response.json()

    if (!data.exists) {
      removeDocument(selectedDocId)
      alert("Este documento ya no está disponible en el servidor.")
      return
    }

    setIsDocumentReady(true)
    setDocId(doc.doc_id)
    resetSummary()
    setSessionId(doc.session_id)
    setFilename(doc.filename)
    setIsHistoryOpen(false)
  }

  console.log(docToDelete);


  return (
    <div className='flex flex-col h-screen bg-surface'>
      <Header onOpenHistory={() => setIsHistoryOpen(true)} />

      <RecentDocuments
        documents={documents}
        active_doc_id={docId || ""}
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        onSelectDocument={handleSelectDocument}
        onRemoveDocument={(doc_id) => setDocToDelete(doc_id)}
      />

      {docToDelete && <ConfirmModal onConfirm={() => { handleRemoveDocument(docToDelete!); setDocToDelete(null) }} onCancel={() => setDocToDelete(null)} />}

      {/* Contenedor */}
      <div className='flex flex-1 mt-16.25 overflow-hidden pb-16 lg:pb-0'>
        {/* Panel izq */}
        <aside className={`w-full lg:flex lg:w-[40%] xl:w-[30%] border-r border-outline-variant flex-col gap-6 bg-surface-container-lowest p-6 overflow-y-auto ${activeTab === 'document' ? "flex" : "hidden"}`}>
          {docId && <HeaderSummary filename={filename} filesize={fileSize} />}
          {!docId && <UploadZone onUploadSuccess={handleUploadSuccess} />}
          {status !== "ready" && docId && <IngestionProgress progress={progress} status={status} filename={filename} />}
          {status === "ready" && <DocumentSummary summary={summary} isDone={isDone} />}
          {status === "ready" && isDone && <ButtonNewDocument onClick={handleNewDocument} />}
        </aside>

        {/* Panel der */}
        <main className={`lg:flex flex-1 overflow-hidden ${activeTab === "chat" ? "flex" : "hidden"}`}>
          <ChatWindow docId={docId} sessionId={sessionId} filename={filename} />
        </main>
      </div>

      {/* Navbar móvil */}
      <nav className='lg:hidden fixed bottom-0 w-full h-16 bg-surface border-t border-outline-variant px-4 py-2 z-50 flex justify-around items-center'>
        <button
          onClick={() => setActiveTab('document')}
          className={`flex-1  cursor-pointer flex flex-col items-center gap-1 transition-all active:opacity-60 text-on-surface-variant hover:text-primary ${activeTab === "document" ? "bg-secondary-container text-on-secondary-container px-6 py-2 rounded-lg" : ""}`}
        >
          <FileText />
          <span className='font-jetbrains text-label-md uppercase tracking-tight'>Documento</span>
        </button>

        <button
          onClick={() => setActiveTab('chat')}
          className={`flex-1  cursor-pointer flex flex-col items-center gap-1 transition-all active:opacity-60 text-on-surface-variant hover:text-primary ${activeTab === "chat" ? "bg-secondary-container text-on-secondary-container px-6 py-2 rounded-lg" : ""}`}
        >
          <MessagesSquare />
          <span className='font-jetbrains text-label-md uppercase tracking-tight'>Chat</span>
        </button>
      </nav>
    </div>
  )
}


export default App
