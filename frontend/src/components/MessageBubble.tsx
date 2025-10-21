'use client'

import { Bot, User, Copy, Check, Info } from 'lucide-react'
import { useState } from 'react'
import { Message } from '@/store/appStore'
import { safeFormatDate } from '@/utils/dateUtils'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface MessageBubbleProps {
  message: Message
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const [copied, setCopied] = useState(false)
  const [showToolInfo, setShowToolInfo] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (error) {
      console.error('Failed to copy text:', error)
    }
  }

  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} animate-slide-up group`}>
      <div className={`flex items-start space-x-3 max-w-3xl ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
        {/* Avatar */}
        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser 
            ? 'bg-blue-500 text-white' 
            : 'bg-gradient-to-r from-blue-500 to-purple-600 text-white'
        }`}>
          {isUser ? <User size={16} /> : <Bot size={16} />}
        </div>

        {/* Message Content */}
        <div className={`flex-1 ${isUser ? 'text-right' : 'text-left'}`}>
          <div className={`inline-block px-4 py-3 rounded-2xl ${
            isUser
              ? 'bg-blue-500 text-white'
              : 'bg-white border border-gray-200 text-gray-900'
          } shadow-sm`}>
            {isUser ? (
              <p className="whitespace-pre-wrap">{message.content}</p>
            ) : (
              <div className="prose prose-sm max-w-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                    ul: ({ children }) => <ul className="mb-2 last:mb-0">{children}</ul>,
                    ol: ({ children }) => <ol className="mb-2 last:mb-0">{children}</ol>,
                    li: ({ children }) => <li className="mb-1">{children}</li>,
                    strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                    em: ({ children }) => <em className="italic">{children}</em>,
                    code: ({ children }) => (
                      <code className="bg-gray-100 text-gray-800 px-1 py-0.5 rounded text-sm">
                        {children}
                      </code>
                    ),
                    pre: ({ children }) => (
                      <pre className="bg-gray-100 text-gray-800 p-3 rounded-lg overflow-x-auto text-sm">
                        {children}
                      </pre>
                    ),
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              </div>
            )}
          </div>

          {/* Timestamp and Actions */}
          <div className={`flex items-center space-x-2 mt-2 text-xs text-gray-500 ${
            isUser ? 'justify-end' : 'justify-start'
          }`}>
            <span>{safeFormatDate(message.timestamp, 'h:mm a', 'Unknown time')}</span>
            
            {!isUser && (
              <>
                {/* Tool Info Button */}
                {message.toolUsed && (
                  <button
                    onClick={() => setShowToolInfo(!showToolInfo)}
                    className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-gray-100 transition-all duration-200"
                    title="Show tool information"
                  >
                    <Info size={12} />
                  </button>
                )}
                
                {/* Copy Button */}
                <button
                  onClick={handleCopy}
                  className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-gray-100 transition-all duration-200"
                  title="Copy message"
                >
                  {copied ? <Check size={12} /> : <Copy size={12} />}
                </button>
              </>
            )}
          </div>

          {/* Tool Information */}
          {!isUser && message.toolUsed && showToolInfo && (
            <div className="mt-2 p-3 bg-blue-50 border border-blue-200 rounded-lg text-xs">
              <div className="flex items-center space-x-2 mb-2">
                <div className={`w-2 h-2 rounded-full ${
                  message.toolUsed === 'RAG' ? 'bg-green-500' : 
                  message.toolUsed === 'Tavily' ? 'bg-blue-500' : 
                  'bg-gray-500'
                }`}></div>
                <span className="font-semibold text-gray-700">
                  {message.toolUsed === 'RAG' ? 'Knowledge Base' : 
                   message.toolUsed === 'Tavily' ? 'Web Search' : 
                   message.toolUsed}
                </span>
              </div>
              <p className="text-gray-600">{message.toolDescription}</p>
            </div>
          )}

          {/* Sources */}
          {message.sources && message.sources.length > 0 && (
            <div className="mt-2 text-xs text-gray-500">
              <p className="font-medium mb-1">Sources:</p>
              <ul className="space-y-1">
                {message.sources.map((source, index) => (
                  <li key={index} className="flex items-center space-x-1">
                    <span className="w-1 h-1 bg-gray-400 rounded-full"></span>
                    <span className="truncate">{source}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
