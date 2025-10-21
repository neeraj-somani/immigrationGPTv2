import axios from 'axios'
import { useAppStore } from '../store/appStore'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

class ApiService {
  private getApiUrl() {
    const settings = useAppStore.getState().settings
    return settings.apiUrl || API_BASE_URL
  }

  async healthCheck() {
    try {
      const response = await axios.get(`${this.getApiUrl()}/health`)
      return response.data
    } catch (error) {
      console.error('Health check failed:', error)
      throw new Error('API service is not available')
    }
  }

  async sendMessage(message: string, conversationId?: string) {
    try {
      const response = await axios.post(`${this.getApiUrl()}/chat`, {
        message,
        conversation_id: conversationId,
      })
      return {
        ...response.data,
        toolUsed: response.data.tool_used,
        toolDescription: response.data.tool_description
      }
    } catch (error) {
      console.error('Send message failed:', error)
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.detail || 'Failed to send message')
      }
      throw new Error('Failed to send message')
    }
  }

  async updateSettings(settings: {
    openai_api_key?: string
    tavily_api_key?: string
    langchain_api_key?: string
  }) {
    try {
      const response = await axios.post(`${this.getApiUrl()}/settings`, settings)
      return response.data
    } catch (error) {
      console.error('Update settings failed:', error)
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.detail || 'Failed to update settings')
      }
      throw new Error('Failed to update settings')
    }
  }

  async getSettingsStatus() {
    try {
      const response = await axios.get(`${this.getApiUrl()}/settings/status`)
      return response.data
    } catch (error) {
      console.error('Get settings status failed:', error)
      throw new Error('Failed to get settings status')
    }
  }
}

export const apiService = new ApiService()
