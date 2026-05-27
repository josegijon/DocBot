import { useState } from 'react';
import { UploadZone } from './components/UploadZone';
import { useIngestionStatus } from './hooks/useIngestionStatus';
import { IngestionProgress } from './components/IngestionProgress';
import { DocumentSummary } from './components/DocumentSummary';
import { useSummary } from './hooks/useSummary';
import { Header } from './components/Header';
import { HeaderSummary } from './components/HeaderSummary';
import { ButtonNewDocument } from './components/ButtonNewDocument';

export const App = () => {
  const [docId, setDocId] = useState<string | null>(null)
  const [filename, setFilename] = useState<string | null>(null)
  const [fileSize, setFileSize] = useState<number>(0)

  const { status, progress } = useIngestionStatus(docId)
  const { summary, isDone } = useSummary(docId, status)

  console.log(docId)
  console.log(status, progress)
  console.log(summary)

  const handleUploadSuccess = (docId: string, filename: string, fileSize: number) => {
    setDocId(docId)
    setFilename(filename)
    setFileSize(fileSize)
  }

  return (
    <div className='flex flex-col h-screen bg-surface'>
      <Header />

      {/* Contenedor */}
      <div className='flex flex-1 mt-16.25 overflow-hidden'>
        {/* Panel izq */}
        <aside className='hidden md:flex md:w-[35%] border-r border-outline-variant flex-col gap-6 bg-surface-container-lowest p-6 overflow-y-auto'>
          {docId && <HeaderSummary filename={filename} filesize={fileSize} />}
          {!docId && <UploadZone onUploadSuccess={handleUploadSuccess} />}
          {status !== "ready" && docId && <IngestionProgress progress={progress} status={status} filename={filename} />}
          {status === "ready" && <DocumentSummary summary={summary} isDone={isDone} />}
          {status === "ready" && isDone && <ButtonNewDocument />}
        </aside>

        {/* Panel der */}
        <main className='flex-1 overflow-hidden'>

        </main>
      </div>

      {/* Navbar móvil */}
      <nav className='md:hidden fixed bottom-0 w-full h-16 bg-surface border-t border-outline-variant flex'>
        <button className="flex-1 text-on-surface cursor-pointer">Documento</button>
        <button className="flex-1 text-on-surface cursor-pointer">Chat</button>
      </nav>
    </div>
  )
}


export default App