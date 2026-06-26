import { FileText, MessagesSquare } from 'lucide-react';
import { toast } from 'sonner';
import { useEffect, useRef, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';

import { useDocumentHistory } from './hooks/useDocumentHistory';
import { useIngestionStatus } from './hooks/useIngestionStatus';
import { useSummary } from './hooks/useSummary';

import { ButtonNewDocument } from './components/ButtonNewDocument';
import { ChatWindow } from './components/ChatWindow';
import { ConfirmModal } from './components/ConfirmModal';
import { DocumentSummary } from './components/DocumentSummary';
import { Header } from './components/Header';
import { HeaderSummary } from './components/HeaderSummary';
import { IngestionProgress } from './components/IngestionProgress';
import { RecentDocuments } from './components/RecentDocuments';
import { UploadZone } from './components/UploadZone';

import { checkDocumentExists, deleteDocument } from './utils/documentApi';
import { NETWORK_ERROR_MESSAGE } from './utils/errorMessages';


const UNNAMED_DOCUMENT_FALLBACK = "Documento sin nombre"

export const App = () => {
  const [docId, setDocId] = useState<string | null>(null)
  const [filename, setFilename] = useState<string | null>(null)
  const [docToDelete, setDocToDelete] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string>(() => uuidv4())
  const [isHistoryOpen, setIsHistoryOpen] = useState(false)
  const [isDocumentReady, setIsDocumentReady] = useState<boolean>(false)
  const [activeTab, setActiveTab] = useState<"document" | "chat">("document")
  const [fileSizeBytes, setFileSizeBytes] = useState<number>(0)

  const { documents, addDocument, removeDocument } = useDocumentHistory()
  const { status, progress, resetStatus } = useIngestionStatus(docId, isDocumentReady)
  const { summary, isDone, resetSummary, error } = useSummary(docId, status)

  const handleUploadSuccess = (docId: string, filename: string, fileSizeBytes: number) => {
    const safeFilename = filename || UNNAMED_DOCUMENT_FALLBACK

    setDocId(docId)
    setFilename(safeFilename)
    setFileSizeBytes(fileSizeBytes)
    addDocument(docId, sessionId, safeFilename)
    setActiveTab('chat')
  }

  const handleNewDocument = () => {
    setIsDocumentReady(false)
    resetStatus()
    setDocId(null)
    setFilename(null)
    setFileSizeBytes(0)
    setSessionId(uuidv4())
    resetSummary()
  }

  const selectDocumentAbortControllerRef = useRef<AbortController | null>(null)

  useEffect(() => {
    return () => {
      selectDocumentAbortControllerRef.current?.abort()
    }
  }, [])

  const handleRemoveDocument = async (doc_id: string) => {
    if (doc_id === docId) handleNewDocument()

    const { success, errorMessage } = await deleteDocument(doc_id)

    if (!success) {
      toast.error(errorMessage ?? NETWORK_ERROR_MESSAGE)
      return
    }

    removeDocument(doc_id)
    toast.success("Documento eliminado")
  }

  useEffect(() => {
    if (status === "ready" && docId && filename) {
      const alreadyExists = documents.some(d => d.doc_id === docId)
      if (!alreadyExists) addDocument(docId, sessionId, filename)
    }
  }, [status, docId, filename, documents, sessionId, addDocument])

  const handleSelectDocument = async (selectedDocId: string) => {
    selectDocumentAbortControllerRef.current?.abort()
    selectDocumentAbortControllerRef.current = null

    if (selectedDocId === docId) {
      setIsHistoryOpen(false)
      return
    }

    const doc = documents.find(d => d.doc_id === selectedDocId)
    if (!doc) return

    selectDocumentAbortControllerRef.current = new AbortController()

    const { exists, errorMessage } = await checkDocumentExists(
      selectedDocId,
      selectDocumentAbortControllerRef.current.signal
    )

    if (errorMessage) {
      toast.error(errorMessage)
      return
    }

    if (exists === null) return

    if (!exists) {
      removeDocument(selectedDocId)
      toast.error("Este documento ya no está disponible en el servidor.")
      return
    }

    setIsDocumentReady(true)
    setDocId(doc.doc_id)
    resetSummary()
    setSessionId(doc.session_id)
    setFilename(doc.filename)
    setIsHistoryOpen(false)
  }

  const shouldShowNewDocumentButton =
    status === "failed" || (status === "ready" && (isDone || error !== null))

  return (
    <div className='flex flex-col h-screen bg-surface relative'>
      <div className='absolute left-0 top-0 bottom-0 w-[calc((100vw-1920px)/2)] bg-surface-container-lowest pointer-events-none'></div>

      <Header
        onOpenHistory={() => setIsHistoryOpen(true)}
        isHistoryOpen={isHistoryOpen}
      />

      <RecentDocuments
        documents={documents}
        active_doc_id={docId}
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        onSelectDocument={handleSelectDocument}
        onRemoveDocument={(doc_id) => { setDocToDelete(doc_id); setIsHistoryOpen(false) }}
      />

      {docToDelete && (
        <ConfirmModal
          title='¿Eliminar este documento?'
          description='Esta acción eliminará el documento y todo su historial de conversación. No se puede deshacer. ¿Quieres continuar?'
          onConfirm={() => { handleRemoveDocument(docToDelete!); setDocToDelete(null) }}
          onCancel={() => setDocToDelete(null)}
        />
      )}

      {/* Contenedor */}
      <div className='flex flex-1 mt-16.25 overflow-hidden pb-16 lg:pb-0 w-full max-w-480 mx-auto'>
        {/* Panel izq */}
        <aside className={`w-full lg:flex lg:w-[40%] xl:w-[30%] border-r border-outline-variant flex-col gap-6 bg-surface-container-lowest p-6 overflow-y-auto ${activeTab === 'document' ? "flex" : "hidden"}`}>
          {docId && filename && <HeaderSummary filename={filename} fileSizeBytes={fileSizeBytes} />}
          {!docId && <UploadZone onUploadSuccess={handleUploadSuccess} />}
          {status !== "ready" && docId && <IngestionProgress progress={progress} status={status} filename={filename} />}
          {status === "ready" && <DocumentSummary summary={summary} isDone={isDone} error={error} />}
          {shouldShowNewDocumentButton && <ButtonNewDocument onClick={handleNewDocument} />}
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
