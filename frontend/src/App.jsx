import React, { useState } from 'react';
import Upload from './components/Upload';
import Chat from './components/Chat';
import './index.css';
function App() {
  const [docId, setDocId] = useState(null);
  const [filename, setFilename] = useState(null);

  return (
    <div className="min-h-screen">
      <header className="bg-black bg-opacity-30 text-white text-center py-12 mb-8">
        <h1 className="text-5xl font-bold mb-2">📚 AskDocs </h1>
        <p className="text-lg opacity-90">Upload documents and ask questions instantly</p>
      </header>

      <main className="max-w-4xl mx-auto px-4 pb-16">
        {!docId ? (
          <Upload onUpload={(id, name) => { setDocId(id); setFilename(name); }} />
        ) : (
          <>
            <div className="bg-white rounded-lg shadow-lg p-6 mb-6 flex justify-between items-center">
              <p className="text-lg font-semibold text-gray-700">📄 <span className="text-purple-600">{filename}</span></p>
              <button 
                onClick={() => { setDocId(null); setFilename(null); }}
                className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg transition"
              >
                📁 Upload Another
              </button>
            </div>
            <Chat docId={docId} />
          </>
        )}
      </main>
    </div>
  );
}

export default App;
