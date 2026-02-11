import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

function Chat({ docId }) {
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setMessages([...messages, { role: 'user', text: question }]);
    setQuestion('');
    setLoading(true);

    try {
      const response = await axios.post('http://127.0.0.1:8000/ask', {
        doc_id: docId,
        question: question,
        top_k: 3,
      });

      setMessages((prev) => [
        ...prev,
        { 
          role: 'assistant', 
          text: response.data.answer, 
          context: response.data.context,
          scores: response.data.similarity_scores
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
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 bg-gray-50">
        {messages.length === 0 && (
          <p className="text-center text-gray-400 text-lg mt-20">Ask me anything about the document! 💬</p>
        )}
        {messages.map((msg, idx) => (
          <div key={idx} className={`mb-4 ${msg.role === 'user' ? 'text-right' : 'text-left'}`}>
            <div className={`inline-block max-w-xs px-4 py-3 rounded-lg ${
              msg.role === 'user' ? 'bg-purple-600 text-white' :
              msg.role === 'error' ? 'bg-red-100 text-red-800' :
              'bg-gray-200 text-gray-800'
            }`}>
              <strong className="block text-sm mb-1 opacity-75">
                {msg.role === 'user' ? '👤 You' : msg.role === 'error' ? '❌ Error' : '🤖 Assistant'}
              </strong>
              <p>{msg.text}</p>
              {msg.context && (
                <details className="mt-2 text-sm cursor-pointer">
                  <summary className="font-bold hover:underline">📖 Sources</summary>
                  <p className="mt-2 whitespace-pre-wrap text-xs">{msg.context}</p>
                </details>
              )}
            </div>
          </div>
        ))}
        {loading && <p className="text-center text-gray-500 italic">⏳ Thinking...</p>}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleAsk} className="p-4 bg-white border-t border-gray-200 flex gap-2">
        <input
          type="text"
          placeholder="Ask a question..."
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
          {loading ? '⏳' : '📤'}
        </button>
      </form>
    </div>
  );
}

export default Chat;
