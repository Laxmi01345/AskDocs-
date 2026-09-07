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
    <div className="flex justify-center items-center">
      <div className="rounded-2xl shadow-2xl p-10 max-w-md w-full" style={{ background: 'rgba(255,255,255,0.95)', backdropFilter: 'blur(10px)' }}>
        <h2 className="text-3xl font-bold text-center mb-2" style={{ color: '#1f2937' }}>
          <span className="mr-2">📤</span>Upload Document
        </h2>
        <p className="text-center mb-8" style={{ color: '#6b7280', fontSize: '0.95em' }}>Supported: TXT, PDF, DOCX</p>

        <form onSubmit={handleUpload}>
          <input
            type="file"
            accept=".txt,.pdf,.docx"
            onChange={handleFileChange}
            disabled={loading}
            className="block w-full mb-6 p-3 rounded-xl cursor-pointer transition"
            style={{ border: '2px solid #e5e7eb', fontSize: '0.9em' }}
          />
          <button 
            type="submit" 
            disabled={loading}
            className="w-full text-white font-bold py-3 rounded-xl transition disabled:opacity-50 disabled:cursor-not-allowed"
            style={{ background: 'linear-gradient(135deg, #3b82f6, #2563eb)' }}
          >
            {loading ? '⏳ Uploading...' : '📁 Upload'}
          </button>
        </form>

        {error && <p className="font-bold mt-4 text-center" style={{ color: '#ef4444' }}>{error}</p>}
      </div>
    </div>
  );
}

export default Upload;
