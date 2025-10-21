import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import '../styles/globals.css'
import { Toaster } from 'react-hot-toast'
import '../utils/storageUtils' // Import storage utilities

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'ImmigrationGPT - AI Immigration Assistant',
  description: 'AI-powered immigration policy assistant with RAG and web search capabilities',
  keywords: ['immigration', 'AI', 'assistant', 'policy', 'USCIS'],
  authors: [{ name: 'ImmigrationGPT Team' }],
}

export const viewport = {
  width: 'device-width',
  initialScale: 1,
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
          {children}
        </div>
        <Toaster 
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#363636',
              color: '#fff',
            },
            success: {
              duration: 3000,
              iconTheme: {
                primary: '#4ade80',
                secondary: '#fff',
              },
            },
            error: {
              duration: 5000,
              iconTheme: {
                primary: '#ef4444',
                secondary: '#fff',
              },
            },
          }}
        />
      </body>
    </html>
  )
}
