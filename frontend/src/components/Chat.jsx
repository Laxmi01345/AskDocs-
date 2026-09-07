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
    <div className="rounded-2xl overflow-hidden flex flex-col h-[500px] shadow-2xl" style={{ background: 'rgba(255,255,255,0.95)', backdropFilter: 'blur(10px)' }}>
      {/* Header */}
      <div className="px-5 py-3 flex justify-between items-center" style={{ background: 'rgba(255,255,255,0.9)', borderBottom: '1px solid rgba(0,0,0,0.06)' }}>
        <span className="text-xs font-medium" style={{ color: '#6b7280' }}>
          {sessionId ? `Session: ${sessionId.slice(0, 8)}...` : 'New conversation'}
        </span>
        <button
          onClick={handleNewChat}
          className="text-xs font-medium hover:underline"
          style={{ color: '#6366f1' }}
        >
          New Chat
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-5 py-4" style={{ background: 'linear-gradient(180deg, #f8f9ff 0%, #f0f1f8 100%)' }}>
        {messages.length === 0 && (
          <p className="text-center mt-20" style={{ color: '#9ca3af', fontSize: '1rem' }}>
            Ask me anything about the document!
          </p>
        )}
        {messages.map((msg, idx) => (
          <div key={idx} className={`mb-4 flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.role === 'assistant' && (
              <div className="mr-2 mt-1 flex-shrink-0">
                <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs" style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', color: 'white' }}>
                  🤖
                </div>
              </div>
            )}
            <div className={`max-w-md ${msg.role === 'user' ? '' : ''}`}>
              <div className="text-xs font-bold mb-1" style={{ color: msg.role === 'user' ? '#3b82f6' : '#6366f1', letterSpacing: '0.05em' }}>
                {msg.role === 'user' ? 'YOU' : msg.role === 'error' ? 'ERROR' : 'ASSISTANT'}
              </div>
              <div
                className="px-4 py-3 rounded-2xl"
                style={
                  msg.role === 'user'
                    ? { background: 'linear-gradient(135deg, #3b82f6, #2563eb)', color: 'white', borderRadius: '18px 18px 4px 18px' }
                    : msg.role === 'error'
                    ? { background: '#fee2e2', color: '#991b1b', borderRadius: '18px 18px 18px 4px' }
                    : { background: 'white', color: '#1f2937', borderRadius: '18px 18px 18px 4px', boxShadow: '0 1px 3px rgba(0,0,0,0.08)', border: '1px solid rgba(0,0,0,0.04)' }
                }
              >
                <div className="prose prose-sm max-w-none">
                  <ReactMarkdown>{msg.text}</ReactMarkdown>
                </div>
                {msg.context && (
                  <details className="mt-3 text-sm cursor-pointer">
                    <summary className="font-bold hover:underline" style={{ color: '#6366f1', fontSize: '0.8em' }}>Sources</summary>
                    <div className="mt-2 space-y-2">
                      {msg.chunks && msg.chunks.map((chunk, ci) => (
                        <div key={ci} className="p-2 rounded-lg" style={{ background: '#f3f4f6', fontSize: '0.75em' }}>
                          <span className="font-semibold">Chunk {chunk.rank}</span>
                          <p className="mt-1" style={{ color: '#6b7280' }}>{chunk.text}</p>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            </div>
            {msg.role === 'user' && (
              <div className="ml-2 mt-1 flex-shrink-0">
                <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs" style={{ background: 'linear-gradient(135deg, #3b82f6, #2563eb)', color: 'white' }}>
                  👤
                </div>
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="flex justify-start mb-4">
            <div className="mr-2 mt-1 flex-shrink-0">
              <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs" style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', color: 'white' }}>
                🤖
              </div>
            </div>
            <div>
              <div className="text-xs font-bold mb-1" style={{ color: '#6366f1', letterSpacing: '0.05em' }}>ASSISTANT</div>
              <div className="px-4 py-3 rounded-2xl" style={{ background: 'white', boxShadow: '0 1px 3px rgba(0,0,0,0.08)', border: '1px solid rgba(0,0,0,0.04)', borderRadius: '18px 18px 18px 4px' }}>
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full animate-bounce" style={{ background: '#6366f1', animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 rounded-full animate-bounce" style={{ background: '#6366f1', animationDelay: '150ms' }}></div>
                  <div className="w-2 h-2 rounded-full animate-bounce" style={{ background: '#6366f1', animationDelay: '300ms' }}></div>
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleAsk} className="p-4 flex gap-2" style={{ background: 'white', borderTop: '1px solid rgba(0,0,0,0.06)' }}>
        <input
          type="text"
          placeholder="Where are..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={loading}
          className="flex-1 px-4 py-3 rounded-xl focus:outline-none focus:ring-2"
          style={{ background: '#f3f4f6', border: '1px solid #e5e7eb', focusRing: '#3b82f6', fontSize: '0.95em' }}
        />
        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="w-10 h-10 rounded-xl flex items-center justify-center transition-all disabled:opacity-40"
          style={{ background: 'linear-gradient(135deg, #3b82f6, #2563eb)', color: 'white' }}
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
        </button>
      </form>
    </div>
  );
}

export default Chat;
