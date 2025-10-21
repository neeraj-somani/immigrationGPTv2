// Utility functions for debugging and fixing localStorage issues
export const clearAppStorage = () => {
  if (typeof window !== 'undefined') {
    localStorage.removeItem('immigration-gpt-storage')
    console.log('Cleared immigration-gpt-storage from localStorage')
  }
}

export const debugAppStorage = (...args: any[]) => {
  if (typeof window !== 'undefined') {
    const stored = localStorage.getItem('immigration-gpt-storage')
    console.log('Current localStorage data:', stored)
    if (args.length > 0 && args[0]) {
      try {
        const parsed = JSON.parse(stored || '{}')
        console.log('Parsed localStorage data:', parsed)
      } catch (error) {
        console.error('Error parsing localStorage data:', error)
      }
    }
  }
}

// Add to window for easy debugging in browser console
if (typeof window !== 'undefined') {
  (window as any).clearAppStorage = clearAppStorage
  (window as any).debugAppStorage = debugAppStorage
}
