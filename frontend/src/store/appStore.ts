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
}

interface AppState {
  // Conversations
  conversations: Conversation[]
  currentConversationId: string | null
  
  // Settings
  settings: Settings
  
  // UI State
  isLoading: boolean
  
  // Actions
  addMessage: (conversationId: string, message: Message) => void
  createConversation: (title: string) => string
  deleteConversation: (id: string) => void
  setCurrentConversation: (id: string | null) => void
  updateSettings: (settings: Partial<Settings>) => void
  setLoading: (loading: boolean) => void
  
  // Getters
  getCurrentConversation: () => Conversation | null
  getConversation: (id: string) => Conversation | null
}

const defaultSettings: Settings = {
  openaiApiKey: '',
  tavilyApiKey: '',
  langchainApiKey: '',
  apiUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
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
          settings: { ...state.settings, ...newSettings },
        }))
      },
      
      setLoading: (loading: boolean) => {
        set({ isLoading: loading })
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
        settings: state.settings,
      }),
      onRehydrateStorage: () => (state) => {
        if (state) {
          try {
            // Ensure all conversations have valid dates when rehydrating from storage
            state.conversations = state.conversations.map(ensureValidDates)
          } catch (error) {
            console.warn('Error rehydrating store:', error)
            // Reset to default state if there's an error
            state.conversations = []
            state.currentConversationId = null
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