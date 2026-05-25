import { useState } from 'react';
import { UploadZone } from './components/UploadZone';
import { useIngestionStatus } from './hooks/useIngestionStatus';
import { IngestionProgress } from './components/IngestionProgress';

export const App = () => {
  const [docId, setDocId] = useState<string | null>(null)
  const [filename, setFilename] = useState<string | null>(null)
  const { status, progress } = useIngestionStatus(docId)

  console.log(docId)
  console.log(status, progress)

  const handleUploadSuccess = (docId: string, filename: string) => {
    setDocId(docId)
    setFilename(filename)
  }

  return (
    <>
      {!docId && <UploadZone onUploadSuccess={handleUploadSuccess} />}
      {docId && <IngestionProgress progress={progress} status={status} filename={filename} />}
    </>
  )
}


export default App