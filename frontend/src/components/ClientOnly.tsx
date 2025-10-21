'use client'

import { useState, useEffect } from 'react'

interface ClientOnlyProps {
  children: React.ReactNode
  fallback?: React.ReactNode
}

export function ClientOnly({ children, fallback = null }: ClientOnlyProps) {
  const [hasMounted, setHasMounted] = useState(false)

  useEffect(() => {
    setHasMounted(true)
  }, [])

  // Always render children once mounted, never go back to fallback
  if (!hasMounted) {
    return <>{fallback}</>
  }

  return <>{children}</>
}
