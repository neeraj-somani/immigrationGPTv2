import { format, isValid, parseISO } from 'date-fns'

/**
 * Safely formats a date value, handling both Date objects and date strings
 * @param date - Date object, date string, or any value
 * @param formatString - Format string for date-fns format function
 * @param fallback - Fallback text if date is invalid
 * @returns Formatted date string or fallback text
 */
export function safeFormatDate(
  date: any,
  formatString: string = 'MMM d, h:mm a',
  fallback: string = 'Unknown date'
): string {
  try {
    // If it's already a valid Date object
    if (date instanceof Date && isValid(date)) {
      return format(date, formatString)
    }
    
    // If it's a string, try to parse it
    if (typeof date === 'string') {
      const parsedDate = parseISO(date)
      if (isValid(parsedDate)) {
        return format(parsedDate, formatString)
      }
    }
    
    // If it's a number (timestamp), try to create a Date
    if (typeof date === 'number') {
      const dateFromTimestamp = new Date(date)
      if (isValid(dateFromTimestamp)) {
        return format(dateFromTimestamp, formatString)
      }
    }
    
    // If all else fails, return fallback
    return fallback
  } catch (error) {
    console.warn('Error formatting date:', error, 'Date value:', date)
    return fallback
  }
}

/**
 * Safely converts a value to a Date object
 * @param value - Any value that might be a date
 * @returns Date object or null if conversion fails
 */
export function safeParseDate(value: any): Date | null {
  try {
    // If it's already a valid Date object
    if (value instanceof Date && isValid(value)) {
      return value
    }
    
    // If it's a string, try to parse it
    if (typeof value === 'string') {
      const parsedDate = parseISO(value)
      if (isValid(parsedDate)) {
        return parsedDate
      }
    }
    
    // If it's a number (timestamp), try to create a Date
    if (typeof value === 'number') {
      const dateFromTimestamp = new Date(value)
      if (isValid(dateFromTimestamp)) {
        return dateFromTimestamp
      }
    }
    
    return null
  } catch (error) {
    console.warn('Error parsing date:', error, 'Value:', value)
    return null
  }
}

/**
 * Ensures a message has a valid Date object for timestamp
 * @param message - Message object
 * @returns Message with valid Date object
 */
export function ensureValidMessageTimestamp(message: any) {
  return {
    ...message,
    timestamp: safeParseDate(message.timestamp) || new Date(),
  }
}

/**
 * Ensures a conversation has valid Date objects for createdAt and updatedAt
 * @param conversation - Conversation object
 * @returns Conversation with valid Date objects
 */
export function ensureValidDates(conversation: any) {
  return {
    ...conversation,
    createdAt: safeParseDate(conversation.createdAt) || new Date(),
    updatedAt: safeParseDate(conversation.updatedAt) || new Date(),
    messages: conversation.messages?.map(ensureValidMessageTimestamp) || [],
  }
}
