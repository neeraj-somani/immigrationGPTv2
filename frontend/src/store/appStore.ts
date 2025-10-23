import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { ensureValidDates } from '@/utils/dateUtils'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  sources?: string[]
  toolUsed?: string
  toolDescription?: string
}

export interface Conversation {
  id: string
  title: string
  messages: Message[]
  createdAt: Date
  updatedAt: Date
}

export interface Settings {
  openaiApiKey: string
  tavilyApiKey: string
  langchainApiKey: string
  apiUrl: string
  retrievalMethod: 'naive' | 'bm25'
}

interface AppState {
  // Conversations
  conversations: Conversation[]
  currentConversationId: string | null
  
  // Settings (non-sensitive only)
  settings: Omit<Settings, 'openaiApiKey' | 'tavilyApiKey' | 'langchainApiKey'>
  
  // UI State
  isLoading: boolean
  
  // Actions
  addMessage: (conversationId: string, message: Message) => void
  createConversation: (title: string) => string
  deleteConversation: (id: string) => void
  setCurrentConversation: (id: string | null) => void
  updateSettings: (settings: Partial<Settings>) => void
  setLoading: (loading: boolean) => void
  
  // Secure API key management
  setApiKeys: (keys: { openaiApiKey?: string; tavilyApiKey?: string; langchainApiKey?: string }) => void
  getApiKeys: () => { openaiApiKey: string; tavilyApiKey: string; langchainApiKey: string }
  clearApiKeys: () => void
  
  // Getters
  getCurrentConversation: () => Conversation | null
  getConversation: (id: string) => Conversation | null
}

const defaultSettings: Omit<Settings, 'openaiApiKey' | 'tavilyApiKey' | 'langchainApiKey'> = {
  apiUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  retrievalMethod: 'bm25', // Default to BM25
}

// Secure API key storage (not persisted)
let secureApiKeys = {
  openaiApiKey: '',
  tavilyApiKey: '',
  langchainApiKey: '',
}

// Clear any existing API keys from localStorage on app start
if (typeof window !== 'undefined') {
  try {
    const stored = localStorage.getItem('immigration-gpt-storage')
    if (stored) {
      const parsed = JSON.parse(stored)
      if (parsed.state?.settings?.openaiApiKey || parsed.state?.settings?.tavilyApiKey || parsed.state?.settings?.langchainApiKey) {
        // Clear API keys from stored data
        parsed.state.settings = {
          ...parsed.state.settings,
          openaiApiKey: '',
          tavilyApiKey: '',
          langchainApiKey: '',
        }
        localStorage.setItem('immigration-gpt-storage', JSON.stringify(parsed))
        console.log('🔒 Cleared API keys from localStorage for security')
      }
    }
  } catch (error) {
    console.warn('Failed to clear API keys from localStorage:', error)
  }
}

// Create the store with proper SSR handling
const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      // Initial state
      conversations: [],
      currentConversationId: null,
      settings: defaultSettings,
      isLoading: false,
      
      // Actions
      addMessage: (conversationId: string, message: Message) => {
        set((state) => ({
          conversations: state.conversations.map((conv) =>
            conv.id === conversationId
              ? {
                  ...conv,
                  messages: [...conv.messages, message],
                  updatedAt: new Date(),
                }
              : conv
          ),
        }))
      },
      
      createConversation: (title: string) => {
        const id = `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
        const newConversation: Conversation = {
          id,
          title,
          messages: [],
          createdAt: new Date(),
          updatedAt: new Date(),
        }
        
        set((state) => ({
          conversations: [newConversation, ...state.conversations],
          currentConversationId: id,
        }))
        
        return id
      },
      
      deleteConversation: (id: string) => {
        set((state) => ({
          conversations: state.conversations.filter((conv) => conv.id !== id),
          currentConversationId: 
            state.currentConversationId === id ? null : state.currentConversationId,
        }))
      },
      
      setCurrentConversation: (id: string | null) => {
        set({ currentConversationId: id })
      },
      
      updateSettings: (newSettings: Partial<Settings>) => {
        set((state) => ({
          settings: { 
            ...state.settings, 
            apiUrl: newSettings.apiUrl || state.settings.apiUrl,
            retrievalMethod: newSettings.retrievalMethod || state.settings.retrievalMethod,
          },
        }))
        
        // Handle API keys separately (not persisted)
        if (newSettings.openaiApiKey !== undefined) {
          secureApiKeys.openaiApiKey = newSettings.openaiApiKey
        }
        if (newSettings.tavilyApiKey !== undefined) {
          secureApiKeys.tavilyApiKey = newSettings.tavilyApiKey
        }
        if (newSettings.langchainApiKey !== undefined) {
          secureApiKeys.langchainApiKey = newSettings.langchainApiKey
        }
      },
      
      setLoading: (loading: boolean) => {
        set({ isLoading: loading })
      },
      
      // Secure API key management
      setApiKeys: (keys: { openaiApiKey?: string; tavilyApiKey?: string; langchainApiKey?: string }) => {
        if (keys.openaiApiKey !== undefined) {
          secureApiKeys.openaiApiKey = keys.openaiApiKey
        }
        if (keys.tavilyApiKey !== undefined) {
          secureApiKeys.tavilyApiKey = keys.tavilyApiKey
        }
        if (keys.langchainApiKey !== undefined) {
          secureApiKeys.langchainApiKey = keys.langchainApiKey
        }
      },
      
      getApiKeys: () => ({
        openaiApiKey: secureApiKeys.openaiApiKey,
        tavilyApiKey: secureApiKeys.tavilyApiKey,
        langchainApiKey: secureApiKeys.langchainApiKey,
      }),
      
      clearApiKeys: () => {
        secureApiKeys = {
          openaiApiKey: '',
          tavilyApiKey: '',
          langchainApiKey: '',
        }
      },
      
      // Getters
      getCurrentConversation: () => {
        const state = get()
        if (!state.currentConversationId) return null
        return state.conversations.find(
          (conv) => conv.id === state.currentConversationId
        ) || null
      },
      
      getConversation: (id: string) => {
        return get().conversations.find((conv) => conv.id === id) || null
      },
    }),
    {
      name: 'immigration-gpt-storage',
      version: 1,
      partialize: (state) => ({
        conversations: state.conversations,
        currentConversationId: state.currentConversationId,
        settings: state.settings, // Only non-sensitive settings are persisted
      }),
      onRehydrateStorage: () => (state) => {
        if (state) {
          try {
            // Ensure all conversations have valid dates when rehydrating from storage
            state.conversations = state.conversations.map(ensureValidDates)
            
            // Ensure settings object is properly initialized with defaults
            state.settings = {
              ...defaultSettings,
              ...state.settings,
            }
          } catch (error) {
            console.warn('Error rehydrating store:', error)
            // Reset to default state if there's an error
            state.conversations = []
            state.currentConversationId = null
            state.settings = defaultSettings
          }
        } else {
          // If no state exists, ensure we have proper defaults
          console.log('No stored state found, using defaults')
        }
      },
    }
  )
)

export { useAppStore }