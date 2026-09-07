import React, { useState } from 'react';
import axios from 'axios';
import API_BASE from '../api';

function Upload({ onUpload }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError(null);
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE}/upload`, formData);
      onUpload(response.data.doc_id, response.data.filename);
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      background: 'white', borderRadius: '16px',
      boxShadow: '0 8px 32px rgba(0,0,0,0.12)',
      padding: '40px', maxWidth: '480px', margin: '0 auto'
    }}>
      <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', textAlign: 'center', color: '#1f2937', marginBottom: '6px' }}>
        📤 Upload Document
      </h2>
      <p style={{ textAlign: 'center', color: '#6b7280', marginBottom: '28px', fontSize: '0.9rem' }}>
        Supported: TXT, PDF, DOCX
      </p>

      <form onSubmit={handleUpload}>
        <input
          type="file"
          accept=".txt,.pdf,.docx"
          onChange={handleFileChange}
          disabled={loading}
          style={{
            display: 'block', width: '100%', marginBottom: '24px',
            padding: '12px 14px', borderRadius: '10px',
            border: '1px solid #d1d5db', background: '#f9fafb',
            fontSize: '0.9rem', boxSizing: 'border-box'
          }}
        />
        <button
          type="submit"
          disabled={loading}
          style={{
            width: '100%', padding: '14px',
            background: 'linear-gradient(135deg, #a855f7, #ec4899)',
            color: 'white', border: 'none', borderRadius: '10px',
            fontSize: '1rem', fontWeight: 'bold', cursor: 'pointer',
            opacity: loading ? 0.5 : 1, transition: 'opacity 0.2s'
          }}
        >
          {loading ? '⏳ Uploading...' : '📁 Upload'}
        </button>
      </form>

      {error && (
        <p style={{ color: '#ef4444', fontWeight: 'bold', textAlign: 'center', marginTop: '16px' }}>
          {error}
        </p>
      )}
    </div>
  );
}

export default Upload;
