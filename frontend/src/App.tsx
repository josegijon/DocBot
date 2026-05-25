import { useState } from 'react';
import { UploadZone } from './components/UploadZone';

export const App = () => {
  const [docId, setDocId] = useState<string | null>(null)

  console.log(docId)

  return (
    <>
      <UploadZone onUploadSuccess={setDocId} />
    </>
  )
}


export default App