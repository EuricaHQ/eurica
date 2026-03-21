'use client'

import { useState } from 'react'

export default function Home() {
  const [message, setMessage] = useState('')
  const [reply, setReply] = useState('')
  const [loading, setLoading] = useState(false)

  async function sendMessage() {
    if (!message.trim()) return

    setLoading(true)
    setReply('')

    try {
      const res = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      })

      const data = await res.json()
      setReply(data.reply ?? 'Keine Antwort')
    } catch (err) {
      setReply('Fehler beim Backend')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen flex flex-col items-center justify-center gap-4 p-6">
      <h1 className="text-2xl font-bold">Eurica Hello World</h1>

      <input
        className="border p-2 w-full max-w-md"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Nachricht eingeben..."
      />

      <button
        onClick={sendMessage}
        className="bg-black text-white px-4 py-2"
      >
        {loading ? 'Sende...' : 'Senden'}
      </button>

      <div className="border p-4 w-full max-w-md min-h-[100px]">
        <strong>Antwort:</strong>
        <div className="mt-2 whitespace-pre-wrap">
          {reply}
        </div>
      </div>
    </main>
  )
}