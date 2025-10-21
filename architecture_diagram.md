# ImmigrationGPT Application Architecture

## System Overview
This diagram shows the complete architecture of the ImmigrationGPT application, including frontend, backend, AI agents, and external services. 

```mermaid
graph TB
    %% Frontend Layer
    subgraph "Frontend Layer (Next.js + React)"
        UI[User Interface]
        CI[ChatInterface Component]
        SM[SettingsModal Component]
        MB[MessageBubble Component]
        TI[TypingIndicator Component]
        AS[App Store - Zustand]
        API_SERVICE[API Service - Axios]
    end

    %% Backend Layer
    subgraph "Backend Layer (FastAPI)"
        FASTAPI[FastAPI Server]
        MAIN[main.py]
        CHAT_ENDPOINT["/chat endpoint"]
        HEALTH_ENDPOINT["/health endpoint"]
        SETTINGS_ENDPOINT["/settings endpoint"]
    end

    %% Core AI System
    subgraph "Core AI System"
        IGPT[ImmigrationGPT Main Class]
        IAS[ImmigrationAgentSystem]
        IRA[ImmigrationRAGAgent]
        IRS[ImmigrationRAGSystem]
    end

    %% Agent Orchestration
    subgraph "Agent Orchestration (LangGraph)"
        SUPERVISOR[Supervisor Agent]
        RAG_AGENT[RAG Agent]
        WEB_AGENT[Web Search Agent]
        STATE_GRAPH[StateGraph Workflow]
    end

    %% Tools and Services
    subgraph "AI Tools & Services"
        RAG_TOOL[RAG Tool]
        TAVILY_TOOL[Tavily Search Tool]
        OPENAI[OpenAI GPT-4o-mini]
        EMBEDDINGS[OpenAI Embeddings]
    end

    %% Vector Store & Data
    subgraph "Vector Store & Data Layer"
        QDRANT[Qdrant Vector Store]
        PDF_LOADER[PyMuPDF Loader]
        TEXT_SPLITTER[RecursiveCharacterTextSplitter]
        DOCUMENTS[Immigration Documents]
    end

    %% External Services
    subgraph "External Services"
        TAVILY_API[Tavily Search API]
        OPENAI_API[OpenAI API]
        USCIS[USCIS.gov]
        IMMIGRATION_FORUM[ImmigrationForum.org]
        MIGRATION_POLICY[MigrationPolicy.org]
    end

    %% User Flow
    UI --> CI
    CI --> API_SERVICE
    API_SERVICE --> FASTAPI
    FASTAPI --> MAIN
    MAIN --> CHAT_ENDPOINT
    CHAT_ENDPOINT --> IGPT
    IGPT --> IAS
    IAS --> STATE_GRAPH
    STATE_GRAPH --> SUPERVISOR
    SUPERVISOR --> RAG_AGENT
    SUPERVISOR --> WEB_AGENT
    RAG_AGENT --> IRA
    WEB_AGENT --> TAVILY_TOOL
    IRA --> IRS
    IRS --> RAG_TOOL
    RAG_TOOL --> QDRANT
    QDRANT --> EMBEDDINGS
    TAVILY_TOOL --> TAVILY_API
    OPENAI --> OPENAI_API
    EMBEDDINGS --> OPENAI_API
    TAVILY_API --> USCIS
    TAVILY_API --> IMMIGRATION_FORUM
    TAVILY_API --> MIGRATION_POLICY
    IRS --> PDF_LOADER
    PDF_LOADER --> DOCUMENTS
    PDF_LOADER --> TEXT_SPLITTER
    TEXT_SPLITTER --> QDRANT

    %% Settings Flow
    SM --> API_SERVICE
    API_SERVICE --> SETTINGS_ENDPOINT
    SETTINGS_ENDPOINT --> MAIN

    %% State Management
    CI --> AS
    SM --> AS
    AS --> API_SERVICE

    %% Styling
    classDef frontend fill:#e1f5fe
    classDef backend fill:#f3e5f5
    classDef ai fill:#fff3e0
    classDef agents fill:#e8f5e8
    classDef tools fill:#fce4ec
    classDef data fill:#f1f8e9
    classDef external fill:#fff8e1

    class UI,CI,SM,MB,TI,AS,API_SERVICE frontend
    class FASTAPI,MAIN,CHAT_ENDPOINT,HEALTH_ENDPOINT,SETTINGS_ENDPOINT backend
    class IGPT,IAS,IRA,IRS ai
    class SUPERVISOR,RAG_AGENT,WEB_AGENT,STATE_GRAPH agents
    class RAG_TOOL,TAVILY_TOOL,OPENAI,EMBEDDINGS tools
    class QDRANT,PDF_LOADER,TEXT_SPLITTER,DOCUMENTS data
    class TAVILY_API,OPENAI_API,USCIS,IMMIGRATION_FORUM,MIGRATION_POLICY external
```

## Technology Stack

### Frontend Technologies
- **Next.js 14** - React framework
- **React 18** - UI library
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations
- **Zustand** - State management
- **Axios** - HTTP client
- **React Markdown** - Markdown rendering
- **React Hot Toast** - Notifications

### Backend Technologies
- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation
- **Python-dotenv** - Environment management

### AI & ML Technologies
- **LangChain** - LLM framework
- **LangGraph** - Agent orchestration
- **OpenAI GPT-4o-mini** - Language model
- **OpenAI Embeddings** - Text embeddings
- **Qdrant** - Vector database
- **Tavily Search** - Web search API

### Document Processing
- **PyMuPDF** - PDF processing
- **RecursiveCharacterTextSplitter** - Text chunking

## Data Flow

1. **User Input**: User types a message in the chat interface
2. **Frontend Processing**: Message is sent via Axios to the FastAPI backend
3. **Backend Routing**: FastAPI routes the request to the chat endpoint
4. **Agent Orchestration**: LangGraph supervisor decides which agent to use
5. **Agent Execution**: Either RAG agent or Web Search agent processes the query
6. **Tool Usage**: Agents use their respective tools (RAG or Tavily search)
7. **Response Generation**: OpenAI generates the final response
8. **Response Return**: Response flows back through the system to the user

## Key Features

- **Multi-Agent System**: RAG agent for document-based queries, Web agent for current information
- **Vector Search**: Qdrant vector store for semantic document retrieval
- **Real-time Chat**: WebSocket-like experience with typing indicators
- **Settings Management**: API key configuration through the UI
- **Conversation History**: Persistent chat history using Zustand
- **Error Handling**: Comprehensive error handling throughout the stack
