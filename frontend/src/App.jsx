import React, { useState } from 'react';
import Upload from './components/Upload';
import Chat from './components/Chat';
import './index.css';

function App() {
  const [docId, setDocId] = useState(null);
  const [filename, setFilename] = useState(null);

  return (
    <div className="min-h-screen">
      <header className="text-center py-10 mb-6">
        <div className="flex items-center justify-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center text-2xl" style={{ background: 'rgba(255,255,255,0.2)' }}>
            📚
          </div>
          <h1 className="text-5xl font-bold text-white">AskDocs</h1>
        </div>
        <p className="text-lg" style={{ color: 'rgba(255,255,255,0.9)' }}>Upload documents and ask questions instantly</p>
      </header>

      <main className="max-w-2xl mx-auto px-4 pb-16">
        {!docId ? (
          <Upload onUpload={(id, name) => { setDocId(id); setFilename(name); }} />
        ) : (
          <>
            <div className="rounded-2xl shadow-lg p-5 mb-5 flex justify-between items-center" style={{ background: 'rgba(255,255,255,0.95)', backdropFilter: 'blur(10px)' }}>
              <p className="text-lg font-semibold" style={{ color: '#1f2937' }}>
                📄 <span style={{ color: '#3b82f6' }}>{filename}</span>
              </p>
              <button
                onClick={() => { setDocId(null); setFilename(null); }}
                className="text-white px-4 py-2 rounded-xl transition font-semibold"
                style={{ background: 'linear-gradient(135deg, #3b82f6, #2563eb)' }}
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
