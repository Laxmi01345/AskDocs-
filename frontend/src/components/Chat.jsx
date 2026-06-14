import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import API_BASE from '../api';

function Chat({ docId }) {
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleNewChat = () => {
    setMessages([]);
    setSessionId(null);
  };

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    const userMessage = question;
    setMessages((prev) => [...prev, { role: 'user', text: userMessage }]);
    setQuestion('');
    setLoading(true);

    try {
      const response = await axios.post(`${API_BASE}/ask`, {
        doc_id: docId,
        question: userMessage,
        top_k: 3,
        session_id: sessionId,
      });

      if (response.data.session_id) {
        setSessionId(response.data.session_id);
      }

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          text: response.data.answer,
          context: response.data.context,
          chunks: response.data.retrieved_chunks,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'error', text: 'Failed to get answer. Please try again.' },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-2xl overflow-hidden flex flex-col h-96">
      <div className="px-4 py-2 bg-purple-50 border-b border-purple-100 flex justify-between items-center">
        <span className="text-xs text-purple-600 font-mono">
          {sessionId ? `Session: ${sessionId.slice(0, 8)}...` : 'New conversation'}
        </span>
        <button
          onClick={handleNewChat}
          className="text-xs text-purple-600 hover:text-purple-800 underline"
        >
          New Chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-6 bg-gray-50">
        {messages.length === 0 && (
          <p className="text-center text-gray-400 text-lg mt-20">Ask me anything about the document!</p>
        )}
        {messages.map((msg, idx) => (
          <div key={idx} className={`mb-4 ${msg.role === 'user' ? 'text-right' : 'text-left'}`}>
            <div className={`inline-block max-w-xs px-4 py-3 rounded-lg ${
              msg.role === 'user' ? 'bg-purple-600 text-white' :
              msg.role === 'error' ? 'bg-red-100 text-red-800' :
              'bg-gray-200 text-gray-800'
            }`}>
              <strong className="block text-sm mb-1 opacity-75">
                {msg.role === 'user' ? 'You' : msg.role === 'error' ? 'Error' : 'Assistant'}
              </strong>
              <div className="prose prose-sm max-w-none">
                <ReactMarkdown>{msg.text}</ReactMarkdown>
              </div>
              {msg.context && (
                <details className="mt-2 text-sm cursor-pointer">
                  <summary className="font-bold hover:underline">Sources</summary>
                  <div className="mt-2 space-y-2">
                    {msg.chunks && msg.chunks.map((chunk, ci) => (
                      <div key={ci} className="p-2 bg-white rounded text-xs">
                        <div className="flex justify-between">
                          <span className="font-semibold">Chunk {chunk.rank}</span>
                        </div>
                        <p className="mt-1 text-gray-600 line-clamp-3">{chunk.text}</p>
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </div>
          </div>
        ))}
        {loading && <p className="text-center text-gray-500 italic">Thinking...</p>}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleAsk} className="p-4 bg-white border-t border-gray-200 flex gap-2">
        <input
          type="text"
          placeholder="Ask a follow-up question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={loading}
          className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-purple-600"
        />
        <button
          type="submit"
          disabled={loading}
          className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-2 rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed font-bold"
        >
          {loading ? '...' : 'Send'}
        </button>
      </form>
    </div>
  );
}

export default Chat;
