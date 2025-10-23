'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, Loader2, Plus, Settings, MessageSquare, Trash2, Zap, Search } from 'lucide-react'
import { useAppStore } from '@/store/appStore'
import { apiService } from '@/services/api'
import { MessageBubble } from './MessageBubble'
import { TypingIndicator } from './TypingIndicator'
import { SettingsModal } from './SettingsModal'
import toast from 'react-hot-toast'

export function ChatInterface() {
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [showConversations, setShowConversations] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const conversationsRef = useRef<HTMLDivElement>(null)
  
  const {
    currentConversationId,
    conversations,
    getCurrentConversation,
    addMessage,
    createConversation,
    deleteConversation,
    setCurrentConversation,
    isLoading,
    setLoading,
    settings,
    getApiKeys,
  } = useAppStore()

  const currentConversation = getCurrentConversation()
  const apiKeys = getApiKeys()

  // Ensure settings object is properly initialized
  const safeSettings = settings || {
    apiUrl: 'http://localhost:8000',
    retrievalMethod: 'bm25' as const,
  }

  // Check if API keys are configured
  const isConfigured = apiKeys.openaiApiKey && apiKeys.tavilyApiKey

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [currentConversation?.messages])

  // Close conversations dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (conversationsRef.current && !conversationsRef.current.contains(event.target as Node)) {
        setShowConversations(false)
      }
    }

    if (showConversations) {
      document.addEventListener('mousedown', handleClickOutside)
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [showConversations])

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return

    const messageText = input.trim()
    setInput('')

    // Create conversation if none exists
    let conversationId = currentConversationId
    if (!conversationId) {
      conversationId = createConversation(`Chat ${Date.now()}`)
    }

    // Add user message
    const userMessage = {
      id: `msg_${Date.now()}_user`,
      role: 'user' as const,
      content: messageText,
      timestamp: new Date(),
    }
    addMessage(conversationId, userMessage)

    // Show typing indicator
    setIsTyping(true)
    setLoading(true)

    try {
      const response = await apiService.sendMessage(messageText, conversationId)
      
      // Add assistant message
      const assistantMessage = {
        id: `msg_${Date.now()}_assistant`,
        role: 'assistant' as const,
        content: response.response,
        timestamp: new Date(),
        sources: response.sources,
        toolUsed: response.toolUsed,
        toolDescription: response.toolDescription,
      }
      addMessage(conversationId, assistantMessage)
      
    } catch (error) {
      console.error('Error sending message:', error)
      toast.error(error instanceof Error ? error.message : 'Failed to send message')
      
      // Add error message
      const errorMessage = {
        id: `msg_${Date.now()}_error`,
        role: 'assistant' as const,
        content: 'Sorry, I encountered an error while processing your request. Please try again.',
        timestamp: new Date(),
      }
      addMessage(conversationId, errorMessage)
    } finally {
      setIsTyping(false)
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const handleNewChat = () => {
    const title = `New Chat ${(conversations?.length || 0) + 1}`
    createConversation(title)
    setShowConversations(false)
  }

  const handleSelectConversation = (id: string) => {
    setCurrentConversation(id)
    setShowConversations(false)
  }

  const handleDeleteConversation = (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    deleteConversation(id)
  }

  // Show setup screen if API keys are not configured
  if (!isConfigured) {
    return (
      <div className="flex-1 flex flex-col h-full">
        {/* Header */}
        <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 px-4 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">ImmigrationGPT</h1>
                <p className="text-sm text-gray-500">AI Immigration Assistant</p>
              </div>
            </div>
          </div>
        </header>

        {/* Setup Content */}
        <div className="flex-1 flex items-center justify-center p-4">
          <div className="text-center max-w-2xl mx-auto">
            <div className="w-20 h-20 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-6">
              <Settings className="w-10 h-10 text-white" />
            </div>
            
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Welcome to ImmigrationGPT
            </h2>
            
            <p className="text-lg text-gray-600 mb-8">
              To get started, you'll need to configure your API keys. This ensures you have access to the latest AI models and search capabilities.
            </p>

            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 mb-8">
              <h3 className="text-lg font-semibold text-yellow-800 mb-3">Required API Keys</h3>
              <div className="space-y-3 text-left">
                <div className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">1</div>
                  <div>
                    <p className="font-medium text-gray-900">OpenAI API Key</p>
                    <p className="text-sm text-gray-600">Required for AI responses and document processing</p>
                    <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-800 text-sm underline">
                      Get your OpenAI API key →
                    </a>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">2</div>
                  <div>
                    <p className="font-medium text-gray-900">Tavily API Key</p>
                    <p className="text-sm text-gray-600">Required for web search and current information</p>
                    <a href="https://tavily.com" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-800 text-sm underline">
                      Get your Tavily API key →
                    </a>
                  </div>
                </div>
              </div>
            </div>

            <button
              onClick={() => setShowSettings(true)}
              className="px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg hover:from-blue-600 hover:to-purple-700 transition-all duration-200 shadow-lg hover:shadow-xl text-lg font-medium"
            >
              Configure API Keys
            </button>

            <p className="text-sm text-gray-500 mt-6">
              Your API keys are stored locally and never shared with our servers.
            </p>
          </div>
        </div>

        {/* Settings Modal */}
        <SettingsModal 
          isOpen={showSettings} 
          onClose={() => setShowSettings(false)}
          isRequired={true}
        />
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col h-full">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">ImmigrationGPT</h1>
              <p className="text-sm text-gray-500">AI Immigration Assistant</p>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <div className="hidden sm:flex items-center space-x-2 px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span>Online</span>
          </div>
          
          {/* Conversations Dropdown */}
          <div className="relative" ref={conversationsRef}>
            <button
              onClick={() => setShowConversations(!showConversations)}
              className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
              title="Conversations"
            >
              <MessageSquare size={20} />
            </button>
            
            {showConversations && (
              <div className="absolute right-0 top-full mt-2 w-80 bg-white rounded-lg shadow-lg border border-gray-200 z-50">
                <div className="p-3 border-b border-gray-200">
                  <button
                    onClick={handleNewChat}
                    className="w-full flex items-center justify-center space-x-2 px-3 py-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg hover:from-blue-600 hover:to-purple-700 transition-all duration-200"
                  >
                    <Plus size={16} />
                    <span className="text-sm font-medium">New Chat</span>
                  </button>
                </div>
                
                <div className="max-h-64 overflow-y-auto">
                  {conversations?.length === 0 ? (
                    <div className="p-4 text-center text-gray-500 text-sm">
                      No conversations yet. Start a new chat!
                    </div>
                  ) : (
                    <div className="p-2 space-y-1">
                      {conversations?.map((conversation) => (
                        <div
                          key={conversation.id}
                          onClick={() => handleSelectConversation(conversation.id)}
                          className={`group relative flex items-center space-x-3 p-2 rounded-lg cursor-pointer transition-all duration-200 ${
                            currentConversationId === conversation.id
                              ? 'bg-blue-50 border border-blue-200'
                              : 'hover:bg-gray-50'
                          }`}
                        >
                          <div className="flex-shrink-0">
                            <MessageSquare 
                              size={14} 
                              className={currentConversationId === conversation.id ? 'text-blue-500' : 'text-gray-400'} 
                            />
                          </div>
                          
                          <div className="flex-1 min-w-0">
                            <p className={`text-sm font-medium truncate ${
                              currentConversationId === conversation.id ? 'text-blue-700' : 'text-gray-900'
                            }`}>
                              {conversation.title}
                            </p>
                            <p className="text-xs text-gray-500">
                              {conversation.messages.length} messages
                            </p>
                          </div>

                          <button
                            onClick={(e) => handleDeleteConversation(e, conversation.id)}
                            className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-100 text-red-500 transition-all duration-200"
                            title="Delete conversation"
                          >
                            <Trash2 size={12} />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
          
          {/* Retrieval Method Indicator */}
          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-1 px-2 py-1 bg-gray-100 rounded-lg">
              {(safeSettings.retrievalMethod || 'bm25') === 'bm25' ? (
                <Zap size={14} className="text-blue-500" />
              ) : (
                <Search size={14} className="text-purple-500" />
              )}
              <span className="text-xs font-medium text-gray-700">
                {(safeSettings.retrievalMethod || 'bm25').toUpperCase()}
              </span>
            </div>
          </div>
          
          <button
            onClick={() => setShowSettings(true)}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
            title="Settings"
          >
            <Settings size={20} />
          </button>
        </div>
      </header>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {currentConversation && currentConversation.messages.length > 0 ? (
          <>
            {currentConversation.messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            
            {isTyping && <TypingIndicator />}
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center max-w-md mx-auto">
              <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <Bot className="w-8 h-8 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                Welcome to ImmigrationGPT
              </h2>
              <p className="text-gray-600 mb-6">
                Ask me anything about US immigration policies, forms, procedures, and more. 
                I can help you understand complex immigration topics with accurate, up-to-date information.
              </p>
              <div className="space-y-4">
                <div className="space-y-2 text-sm text-gray-500">
                  <p className="font-medium text-gray-700">💡 Try asking:</p>
                  <div className="grid gap-2">
                    <button
                      onClick={() => setInput("What is Form I-730?")}
                      className="text-left p-3 bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors border border-gray-200 hover:border-gray-300"
                    >
                      <span className="text-gray-700 font-medium">What is Form I-730?</span>
                    </button>
                    <button
                      onClick={() => setInput("How do I apply for asylum?")}
                      className="text-left p-3 bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors border border-gray-200 hover:border-gray-300"
                    >
                      <span className="text-gray-700 font-medium">How do I apply for asylum?</span>
                    </button>
                    <button
                      onClick={() => setInput("What are the requirements for refugee status?")}
                      className="text-left p-3 bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors border border-gray-200 hover:border-gray-300"
                    >
                      <span className="text-gray-700 font-medium">What are the requirements for refugee status?</span>
                    </button>
                    <button
                      onClick={() => setInput("What documents do I need for a green card application?")}
                      className="text-left p-3 bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors border border-gray-200 hover:border-gray-300"
                    >
                      <span className="text-gray-700 font-medium">What documents do I need for a green card application?</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 bg-white/80 backdrop-blur-sm p-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-end space-x-3">
            <div className="flex-1">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask me about US immigration policies..."
                className="w-full px-4 py-3 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows={1}
                style={{ minHeight: '48px', maxHeight: '120px' }}
                disabled={isLoading}
              />
            </div>
            <button
              onClick={handleSendMessage}
              disabled={!input.trim() || isLoading}
              className="px-4 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Settings Modal */}
      <SettingsModal 
        isOpen={showSettings} 
        onClose={() => setShowSettings(false)}
        isRequired={false}
      />
    </div>
  )
}
