import { useState, useRef, useEffect } from 'react'
import './App.css'

// Change this if your backend runs somewhere other than localhost:8000
const API_URL = 'http://localhost:8000/chat'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const messagesEndRef = useRef(null)

  // Persist session_id in localStorage so a page refresh doesn't lose
  // the conversation - the backend already remembers it via SQLite,
  // this just lets the browser reconnect to the same session.
  useEffect(() => {
    const saved = localStorage.getItem('chatbot_session_id')
    if (saved) setSessionId(saved)
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  const sendMessage = async () => {
    const text = input.trim()
    if (!text || isLoading) return

    setMessages(prev => [...prev, { role: 'user', content: text }])
    setInput('')
    setIsLoading(true)

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      })

      if (!response.ok) throw new Error('Request failed')

      const data = await response.json()

      if (!sessionId) {
        setSessionId(data.session_id)
        localStorage.setItem('chatbot_session_id', data.session_id)
      }

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.reply,
        route: data.route,
      }])
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: "I'm having trouble connecting right now. Please check that the backend is running and try again.",
        route: 'error',
      }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-mark" aria-hidden="true" />
        <div>
          <h1>Mitra</h1>
          <p className="header-sub">a quiet space to talk things through</p>
        </div>
      </header>

      <main className="chat-window">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>Hi, I'm here to listen.</p>
            <p className="empty-sub">Whatever's on your mind, you can start wherever feels right.</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`message-row ${msg.role === 'user' ? 'from-user' : 'from-bot'}`}
          >
            <div className={`bubble ${msg.role === 'user' ? 'bubble-user' : 'bubble-bot'} ${msg.route === 'crisis' ? 'bubble-crisis' : ''}`}>
              {msg.route === 'crisis' && (
                <span className="crisis-label">Support resource</span>
              )}
              {msg.content}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="message-row from-bot">
            <div className="bubble bubble-bot bubble-loading">
              <span className="breathing-dot" />
              <span className="breathing-dot" />
              <span className="breathing-dot" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </main>

      <footer className="input-bar">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type here..."
          rows={1}
        />
        <button onClick={sendMessage} disabled={isLoading || !input.trim()}>
          Send
        </button>
      </footer>
    </div>
  )
}

export default App