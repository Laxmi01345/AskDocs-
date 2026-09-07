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
    <div style={{
      background: 'white', borderRadius: '16px',
      boxShadow: '0 8px 32px rgba(0,0,0,0.12)',
      overflow: 'hidden', display: 'flex', flexDirection: 'column',
      height: '520px'
    }}>
      {/* Header */}
      <div style={{
        padding: '12px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        borderBottom: '1px solid #f3f4f6'
      }}>
        <span style={{ fontSize: '0.8rem', color: '#9ca3af' }}>
          {sessionId ? `Session: ${sessionId.slice(0, 8)}...` : 'New conversation'}
        </span>
        <button
          onClick={handleNewChat}
          style={{ fontSize: '0.8rem', color: '#7c3aed', background: 'none', border: 'none', cursor: 'pointer', textDecoration: 'underline' }}
        >
          New Chat
        </button>
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '20px', background: '#fafafa' }}>
        {messages.length === 0 && (
          <p style={{ textAlign: 'center', color: '#9ca3af', marginTop: '160px' }}>
            Ask me anything about the document!
          </p>
        )}
        {messages.map((msg, idx) => (
          <div key={idx} style={{
            marginBottom: '16px',
            display: 'flex',
            justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
            alignItems: 'flex-start', gap: '8px'
          }}>
            {msg.role !== 'user' && (
              <div style={{
                width: '28px', height: '28px', borderRadius: '50%', flexShrink: 0,
                background: 'linear-gradient(135deg, #a855f7, #7c3aed)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '14px', marginTop: '18px'
              }}>🤖</div>
            )}
            <div style={{ maxWidth: '70%' }}>
              <div style={{
                fontSize: '0.7rem', fontWeight: 'bold', marginBottom: '4px',
                color: msg.role === 'user' ? '#3b82f6' : '#7c3aed',
                letterSpacing: '0.05em', textAlign: msg.role === 'user' ? 'right' : 'left'
              }}>
                {msg.role === 'user' ? 'YOU' : msg.role === 'error' ? 'ERROR' : 'ASSISTANT'}
              </div>
              <div style={{
                padding: '12px 16px',
                borderRadius: msg.role === 'user' ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
                background: msg.role === 'user'
                  ? 'linear-gradient(135deg, #3b82f6, #2563eb)'
                  : msg.role === 'error' ? '#fee2e2' : 'white',
                color: msg.role === 'user' ? 'white' : msg.role === 'error' ? '#991b1b' : '#1f2937',
                boxShadow: msg.role === 'user' ? 'none' : '0 1px 4px rgba(0,0,0,0.06)',
                border: msg.role === 'user' ? 'none' : '1px solid #f3f4f6',
                fontSize: '0.9rem', lineHeight: '1.6'
              }}>
                <div style={{ fontSize: '0.85rem' }}>
                  <ReactMarkdown>{msg.text}</ReactMarkdown>
                </div>
                {msg.context && (
                  <details style={{ marginTop: '10px', fontSize: '0.8rem' }}>
                    <summary style={{ cursor: 'pointer', fontWeight: 'bold', color: '#7c3aed' }}>Sources</summary>
                    <div style={{ marginTop: '8px' }}>
                      {msg.chunks && msg.chunks.map((chunk, ci) => (
                        <div key={ci} style={{ padding: '8px', background: '#f3f4f6', borderRadius: '6px', marginBottom: '6px', fontSize: '0.75rem' }}>
                          <span style={{ fontWeight: '600' }}>Chunk {chunk.rank}</span>
                          <p style={{ color: '#6b7280', marginTop: '4px' }}>{chunk.text}</p>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            </div>
            {msg.role === 'user' && (
              <div style={{
                width: '28px', height: '28px', borderRadius: '50%', flexShrink: 0,
                background: 'linear-gradient(135deg, #3b82f6, #2563eb)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '14px', marginTop: '18px'
              }}>👤</div>
            )}
          </div>
        ))}
        {loading && (
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', marginBottom: '16px' }}>
            <div style={{
              width: '28px', height: '28px', borderRadius: '50%', flexShrink: 0,
              background: 'linear-gradient(135deg, #a855f7, #7c3aed)',
              display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '14px'
            }}>🤖</div>
            <div>
              <div style={{ fontSize: '0.7rem', fontWeight: 'bold', color: '#7c3aed', marginBottom: '4px', letterSpacing: '0.05em' }}>ASSISTANT</div>
              <div style={{
                padding: '12px 16px', borderRadius: '16px 16px 16px 4px',
                background: 'white', boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
                border: '1px solid #f3f4f6'
              }}>
                <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
                  <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#a855f7', animation: 'bounce 1s infinite 0ms' }}></div>
                  <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#a855f7', animation: 'bounce 1s infinite 150ms' }}></div>
                  <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#a855f7', animation: 'bounce 1s infinite 300ms' }}></div>
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleAsk} style={{
        padding: '12px 16px', display: 'flex', gap: '8px',
        borderTop: '1px solid #f3f4f6', background: 'white'
      }}>
        <input
          type="text"
          placeholder="Where are..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={loading}
          style={{
            flex: 1, padding: '12px 16px', borderRadius: '12px',
            border: '1px solid #e5e7eb', background: '#f3f4f6',
            fontSize: '0.9rem', outline: 'none'
          }}
        />
        <button
          type="submit"
          disabled={loading || !question.trim()}
          style={{
            width: '42px', height: '42px', borderRadius: '12px', border: 'none',
            background: 'linear-gradient(135deg, #3b82f6, #2563eb)',
            color: 'white', cursor: 'pointer', display: 'flex',
            alignItems: 'center', justifyContent: 'center',
            opacity: loading || !question.trim() ? 0.4 : 1
          }}
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
        </button>
      </form>

      <style>{`
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); }
          40% { transform: translateY(-6px); }
        }
      `}</style>
    </div>
  );
}

export default Chat;
