import { ChatInterface } from '@/components/ChatInterface'
import { ClientOnly } from '@/components/ClientOnly'

export default function Home() {
  return (
    <main className="h-screen flex flex-col">
      {/* Chat Interface - Client only for hydration */}
      <ClientOnly fallback={
        <div className="flex-1 flex flex-col h-full">
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center max-w-md mx-auto">
                <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4 animate-pulse"></div>
                <div className="h-8 w-64 bg-gray-200 rounded animate-pulse mx-auto mb-2"></div>
                <div className="h-4 w-96 bg-gray-200 rounded animate-pulse mx-auto mb-6"></div>
                <div className="space-y-2">
                  <div className="h-4 w-24 bg-gray-200 rounded animate-pulse mx-auto"></div>
                  <div className="h-3 w-32 bg-gray-200 rounded animate-pulse mx-auto"></div>
                  <div className="h-3 w-28 bg-gray-200 rounded animate-pulse mx-auto"></div>
                  <div className="h-3 w-36 bg-gray-200 rounded animate-pulse mx-auto"></div>
                </div>
              </div>
            </div>
          </div>
          <div className="border-t border-gray-200 bg-white/80 backdrop-blur-sm p-4">
            <div className="max-w-4xl mx-auto">
              <div className="flex items-end space-x-3">
                <div className="flex-1">
                  <div className="w-full h-12 bg-gray-200 rounded-lg animate-pulse"></div>
                </div>
                <div className="w-12 h-12 bg-gray-200 rounded-lg animate-pulse"></div>
              </div>
            </div>
          </div>
        </div>
      }>
        <ChatInterface />
      </ClientOnly>
    </main>
  )
}
