import { useState } from 'react';
import { UploadZone } from './components/UploadZone';
import { useIngestionStatus } from './hooks/useIngestionStatus';
import { IngestionProgress } from './components/IngestionProgress';
import { DocumentSummary } from './components/DocumentSummary';
import { useSummary } from './hooks/useSummary';

export const App = () => {
  const [docId, setDocId] = useState<string | null>(null)
  const [filename, setFilename] = useState<string | null>(null)

  const { status, progress } = useIngestionStatus(docId)
  const { summary, isDone } = useSummary(docId, status)

  console.log(docId)
  console.log(status, progress)
  console.log(summary)

  const handleUploadSuccess = (docId: string, filename: string) => {
    setDocId(docId)
    setFilename(filename)
  }

  return (
    <>
      {!docId && <UploadZone onUploadSuccess={handleUploadSuccess} />}
      {status !== "ready" && docId && <IngestionProgress progress={progress} status={status} filename={filename} />}
      {status === "ready" && <DocumentSummary summary={summary} isDone={isDone} />}
    </>
  )
}


export default App