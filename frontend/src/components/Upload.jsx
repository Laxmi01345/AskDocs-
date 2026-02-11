import React, { useState } from 'react';
import axios from 'axios';

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
      const response = await axios.post('http://127.0.0.1:8000/upload', formData);
      onUpload(response.data.doc_id, response.data.filename);
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex justify-center items-center">
      <div className="bg-white rounded-xl shadow-2xl p-10 max-w-md w-full">
        <h2 className="text-3xl font-bold text-center mb-2 text-gray-800">📤 Upload Document</h2>
        <p className="text-center text-gray-600 mb-8">Supported: TXT, PDF, DOCX</p>

        <form onSubmit={handleUpload}>
          <input
            type="file"
            accept=".txt,.pdf,.docx"
            onChange={handleFileChange}
            disabled={loading}
            className="block w-full mb-6 p-3 border-2 border-gray-300 rounded-lg cursor-pointer hover:border-purple-600 transition"
          />
          <button 
            type="submit" 
            disabled={loading}
            className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-bold py-3 rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? '⏳ Uploading...' : '📁 Upload'}
          </button>
        </form>

        {error && <p className="text-red-600 font-bold mt-4 text-center">{error}</p>}
      </div>
    </div>
  );
}

export default Upload;
