'use client'

import { useState, useEffect } from 'react'
import { X, Save, Eye, EyeOff, CheckCircle, AlertCircle } from 'lucide-react'
import { useAppStore } from '@/store/appStore'
import { apiService } from '@/services/api'
import toast from 'react-hot-toast'

interface SettingsModalProps {
  isOpen: boolean
  onClose: () => void
  isRequired?: boolean // If true, modal cannot be closed without valid API keys
}

export function SettingsModal({ isOpen, onClose, isRequired = false }: SettingsModalProps) {
  const { settings, updateSettings, getApiKeys, setApiKeys } = useAppStore()
  const apiKeys = getApiKeys()
  const [formData, setFormData] = useState({
    openaiApiKey: '',
    tavilyApiKey: '',
    langchainApiKey: '',
    retrievalMethod: 'bm25' as 'naive' | 'bm25',
  })
  const [showKeys, setShowKeys] = useState({
    openai: false,
    tavily: false,
    langchain: false,
  })
  const [isSaving, setIsSaving] = useState(false)
  const [isValidating, setIsValidating] = useState(false)
  const [validationResults, setValidationResults] = useState<{
    openai: boolean | null
    tavily: boolean | null
  }>({ openai: null, tavily: null })
  const [settingsStatus, setSettingsStatus] = useState<any>(null)

  useEffect(() => {
    if (isOpen) {
      setFormData({
        openaiApiKey: apiKeys.openaiApiKey,
        tavilyApiKey: apiKeys.tavilyApiKey,
        langchainApiKey: apiKeys.langchainApiKey,
        retrievalMethod: settings.retrievalMethod,
      })
      loadSettingsStatus()
    }
  }, [isOpen]) // Remove apiKeys and settings from dependencies to prevent infinite re-renders

  // Update form data when API keys change (but only if modal is open)
  useEffect(() => {
    if (isOpen) {
      setFormData(prev => ({
        ...prev,
        openaiApiKey: apiKeys.openaiApiKey,
        tavilyApiKey: apiKeys.tavilyApiKey,
        langchainApiKey: apiKeys.langchainApiKey,
        retrievalMethod: settings.retrievalMethod,
      }))
    }
  }, [apiKeys.openaiApiKey, apiKeys.tavilyApiKey, apiKeys.langchainApiKey, settings.retrievalMethod, isOpen])

  const loadSettingsStatus = async () => {
    try {
      const status = await apiService.getSettingsStatus()
      setSettingsStatus(status)
    } catch (error) {
      console.error('Failed to load settings status:', error)
    }
  }

  const validateApiKeys = async () => {
    if (!formData.openaiApiKey || !formData.tavilyApiKey) {
      toast.error('Please enter both OpenAI and Tavily API keys')
      return
    }

    setIsValidating(true)
    try {
      // Update backend with current keys for validation
      await apiService.updateSettings({
        openai_api_key: formData.openaiApiKey,
        tavily_api_key: formData.tavilyApiKey,
        langchain_api_key: formData.langchainApiKey,
      })

      // Check status after update
      const status = await apiService.getSettingsStatus()
      setSettingsStatus(status)
      
      // Update validation results based on status
      setValidationResults({
        openai: status.openai_api_key?.includes('✅'),
        tavily: status.tavily_api_key?.includes('✅')
      })

      if (status.service_status?.includes('✅')) {
        toast.success('API keys validated successfully!')
      } else {
        toast.error('API key validation failed. Please check your keys.')
      }
    } catch (error) {
      console.error('API key validation failed:', error)
      toast.error('Failed to validate API keys. Please check your keys and try again.')
      setValidationResults({ openai: false, tavily: false })
    } finally {
      setIsValidating(false)
    }
  }

  const handleSave = async () => {
    console.log('Saving settings with formData:', formData)
    
    // Validate required fields
    if (!formData.openaiApiKey.trim()) {
      toast.error('OpenAI API Key is required')
      return
    }
    if (!formData.tavilyApiKey.trim()) {
      toast.error('Tavily API Key is required')
      return
    }

    setIsSaving(true)
    try {
      // Update API keys securely (not persisted)
      setApiKeys({
        openaiApiKey: formData.openaiApiKey,
        tavilyApiKey: formData.tavilyApiKey,
        langchainApiKey: formData.langchainApiKey,
      })

      // Update non-sensitive settings (persisted)
      updateSettings({
        retrievalMethod: formData.retrievalMethod,
      })

      // Update backend settings
      await apiService.updateSettings({
        openai_api_key: formData.openaiApiKey,
        tavily_api_key: formData.tavilyApiKey,
        langchain_api_key: formData.langchainApiKey,
      })

      toast.success('Settings saved successfully!')
      onClose()
    } catch (error) {
      console.error('Failed to save settings:', error)
      toast.error(error instanceof Error ? error.message : 'Failed to save settings')
    } finally {
      setIsSaving(false)
    }
  }

  const handleClose = () => {
    if (isRequired && (!formData.openaiApiKey || !formData.tavilyApiKey)) {
      toast.error('Please configure your API keys before continuing')
      return
    }
    onClose()
  }

  const toggleKeyVisibility = (key: 'openai' | 'tavily' | 'langchain') => {
    setShowKeys(prev => ({ ...prev, [key]: !prev[key] }))
  }

  const handleRetrievalMethodChange = async (method: 'naive' | 'bm25') => {
    try {
      await apiService.setRetrievalMethod(method)
      setFormData(prev => ({ ...prev, retrievalMethod: method }))
      updateSettings({ retrievalMethod: method })
      toast.success(`Retrieval method switched to ${method.toUpperCase()}`)
      
      // Reload status to get updated information
      await loadSettingsStatus()
    } catch (error) {
      console.error('Failed to change retrieval method:', error)
      toast.error(error instanceof Error ? error.message : 'Failed to change retrieval method')
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h2 className="text-xl font-bold text-gray-900">Settings</h2>
          {!isRequired && (
            <button
              onClick={handleClose}
              className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <X size={20} />
            </button>
          )}
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
          <div className="space-y-6">
            {/* API Status */}
            {settingsStatus && (
              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="font-semibold text-gray-900 mb-3">Service Status</h3>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">OpenAI API</span>
                    <div className="flex items-center space-x-2">
                      {settingsStatus.openai_api_key?.includes('✅') ? (
                        <CheckCircle size={16} className="text-green-500" />
                      ) : (
                        <AlertCircle size={16} className="text-red-500" />
                      )}
                      <span className="text-sm">{settingsStatus.openai_api_key}</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Tavily API</span>
                    <div className="flex items-center space-x-2">
                      {settingsStatus.tavily_api_key?.includes('✅') ? (
                        <CheckCircle size={16} className="text-green-500" />
                      ) : (
                        <AlertCircle size={16} className="text-red-500" />
                      )}
                      <span className="text-sm">{settingsStatus.tavily_api_key}</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Service Status</span>
                    <div className="flex items-center space-x-2">
                      {settingsStatus.service_status?.includes('✅') ? (
                        <CheckCircle size={16} className="text-green-500" />
                      ) : (
                        <AlertCircle size={16} className="text-red-500" />
                      )}
                      <span className="text-sm">{settingsStatus.service_status}</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* API Keys */}
            <div className="space-y-4">
              <h3 className="font-semibold text-gray-900">API Keys</h3>
              
              {/* OpenAI API Key */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  OpenAI API Key *
                </label>
                <div className="relative">
                  <input
                    type={showKeys.openai ? 'text' : 'password'}
                    value={formData.openaiApiKey}
                    onChange={(e) => {
                      console.log('OpenAI API key changed:', e.target.value)
                      setFormData(prev => ({ ...prev, openaiApiKey: e.target.value }))
                    }}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 pr-10"
                    placeholder="sk-..."
                  />
                  <button
                    type="button"
                    onClick={() => toggleKeyVisibility('openai')}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
                  >
                    {showKeys.openai ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>

              {/* Tavily API Key */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Tavily API Key *
                </label>
                <div className="relative">
                  <input
                    type={showKeys.tavily ? 'text' : 'password'}
                    value={formData.tavilyApiKey}
                    onChange={(e) => setFormData(prev => ({ ...prev, tavilyApiKey: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 pr-10"
                    placeholder="tvly-..."
                  />
                  <button
                    type="button"
                    onClick={() => toggleKeyVisibility('tavily')}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
                  >
                    {showKeys.tavily ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>

              {/* LangChain API Key */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  LangChain API Key (Optional)
                </label>
                <div className="relative">
                  <input
                    type={showKeys.langchain ? 'text' : 'password'}
                    value={formData.langchainApiKey}
                    onChange={(e) => setFormData(prev => ({ ...prev, langchainApiKey: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 pr-10"
                    placeholder="ls__..."
                  />
                  <button
                    type="button"
                    onClick={() => toggleKeyVisibility('langchain')}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
                  >
                    {showKeys.langchain ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Used for tracing and monitoring (optional)
                </p>
              </div>

            </div>

            {/* Retrieval Method Selection */}
            <div className="space-y-4">
              <h3 className="font-semibold text-gray-900">Retrieval Method</h3>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Choose Retrieval Method
                </label>
                <div className="flex space-x-4">
                  <button
                    onClick={() => handleRetrievalMethodChange('bm25')}
                    disabled={!settingsStatus?.service_status?.includes('✅')}
                    className={`px-4 py-2 rounded-lg border transition-colors flex-1 ${
                      formData.retrievalMethod === 'bm25'
                        ? 'bg-blue-500 text-white border-blue-500'
                        : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                    } ${!settingsStatus?.service_status?.includes('✅') ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    <div className="text-center">
                      <div className="font-medium">BM25</div>
                      <div className="text-xs opacity-75">Fast & Accurate</div>
                    </div>
                  </button>
                  
                  <button
                    onClick={() => handleRetrievalMethodChange('naive')}
                    disabled={!settingsStatus?.service_status?.includes('✅')}
                    className={`px-4 py-2 rounded-lg border transition-colors flex-1 ${
                      formData.retrievalMethod === 'naive'
                        ? 'bg-blue-500 text-white border-blue-500'
                        : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                    } ${!settingsStatus?.service_status?.includes('✅') ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    <div className="text-center">
                      <div className="font-medium">Naive</div>
                      <div className="text-xs opacity-75">Semantic Search</div>
                    </div>
                  </button>
                </div>
                
                {/* Current Status */}
                {settingsStatus && (
                  <div className="mt-3 p-3 bg-gray-50 rounded-lg">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600">Current Method:</span>
                      <span className="font-medium text-gray-900">
                        {settingsStatus.retrieval_method?.toUpperCase() || 'Unknown'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-sm mt-1">
                      <span className="text-gray-600">BM25 Available:</span>
                      <span className={`font-medium ${settingsStatus.bm25_available ? 'text-green-600' : 'text-red-600'}`}>
                        {settingsStatus.bm25_available ? '✅ Yes' : '❌ No'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-sm mt-1">
                      <span className="text-gray-600">Naive Available:</span>
                      <span className={`font-medium ${settingsStatus.naive_available ? 'text-green-600' : 'text-red-600'}`}>
                        {settingsStatus.naive_available ? '✅ Yes' : '❌ No'}
                      </span>
                    </div>
                  </div>
                )}
                
                <p className="text-xs text-gray-500 mt-2">
                  BM25 is recommended for most queries (faster, more accurate). 
                  Use Naive for complex conceptual questions.
                </p>
              </div>
            </div>
            <div className="bg-blue-50 rounded-lg p-4">
              <h4 className="font-medium text-blue-900 mb-2">Getting API Keys</h4>
              <div className="text-sm text-blue-800 space-y-1">
                <p>• <strong>OpenAI:</strong> Get your API key from <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" className="underline">platform.openai.com</a></p>
                <p>• <strong>Tavily:</strong> Get your API key from <a href="https://tavily.com" target="_blank" rel="noopener noreferrer" className="underline">tavily.com</a></p>
                <p>• <strong>LangChain:</strong> Get your API key from <a href="https://smith.langchain.com" target="_blank" rel="noopener noreferrer" className="underline">smith.langchain.com</a></p>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-6 border-t border-gray-200">
          <div className="flex items-center space-x-3">
            <button
              onClick={validateApiKeys}
              disabled={isValidating || !formData.openaiApiKey || !formData.tavilyApiKey}
              className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
            >
              {isValidating ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Validating...</span>
                </>
              ) : (
                <>
                  <CheckCircle size={16} />
                  <span>Validate Keys</span>
                </>
              )}
            </button>
          </div>
          
          <div className="flex items-center space-x-3">
            {!isRequired && (
              <button
                onClick={handleClose}
                className="px-4 py-2 text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
            )}
            <button
              onClick={handleSave}
              disabled={isSaving || !formData.openaiApiKey || !formData.tavilyApiKey}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
            >
              {isSaving ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Saving...</span>
                </>
              ) : (
                <>
                  <Save size={16} />
                  <span>Save Settings</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}