'use client'

import { Bot } from 'lucide-react'

export function TypingIndicator() {
  return (
    <div className="flex justify-start animate-slide-up">
      <div className="flex items-start space-x-3 max-w-3xl">
        {/* Avatar */}
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 text-white flex items-center justify-center">
          <Bot size={16} />
        </div>

        {/* Typing Animation */}
        <div className="flex-1">
          <div className="inline-block px-4 py-3 bg-white border border-gray-200 rounded-2xl shadow-sm">
            <div className="flex items-center space-x-1">
              <span className="text-sm text-gray-500 mr-2">ImmigrationGPT is typing</span>
              <div className="typing-indicator">
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
