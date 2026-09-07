import React, { useState } from 'react';
import Upload from './components/Upload';
import Chat from './components/Chat';
import './index.css';

function App() {
  const [docId, setDocId] = useState(null);
  const [filename, setFilename] = useState(null);

  return (
    <div>
      <header style={{ padding: '40px 20px 35px', textAlign: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px', marginBottom: '8px' }}>
          <div style={{
            width: '44px', height: '44px', borderRadius: '10px',
            background: 'linear-gradient(135deg, #22c55e, #3b82f6)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '24px', boxShadow: '0 2px 8px rgba(0,0,0,0.15)'
          }}>📚</div>
          <h1 style={{ fontSize: '3rem', fontWeight: 'bold', color: 'white', margin: 0 }}>AskDocs</h1>
        </div>
        <p style={{ color: 'rgba(255,255,255,0.9)', fontSize: '1.1rem', margin: 0 }}>
          Upload documents and ask questions instantly
        </p>
      </header>

      <main style={{ maxWidth: '520px', margin: '0 auto', padding: '0 20px 60px' }}>
        {!docId ? (
          <Upload onUpload={(id, name) => { setDocId(id); setFilename(name); }} />
        ) : (
          <>
            <div style={{
              background: 'white', borderRadius: '16px', boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
              padding: '18px 24px', marginBottom: '20px',
              display: 'flex', justifyContent: 'space-between', alignItems: 'center'
            }}>
              <p style={{ fontSize: '1rem', fontWeight: '600', color: '#374151', margin: 0 }}>
                📄 <span style={{ color: '#7c3aed' }}>{filename}</span>
              </p>
              <button
                onClick={() => { setDocId(null); setFilename(null); }}
                style={{
                  background: 'linear-gradient(135deg, #a855f7, #ec4899)',
                  color: 'white', border: 'none', padding: '10px 20px',
                  borderRadius: '10px', cursor: 'pointer', fontWeight: '600',
                  fontSize: '0.9rem', transition: 'opacity 0.2s'
                }}
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
