# Import dependencies
import os
import warnings
from uuid import uuid4
from typing import TypedDict, Annotated, List, Optional
import operator
import functools

warnings.filterwarnings('ignore')

# LangChain imports
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

# LangGraph imports
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

# Vector store imports
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient

# Tavily search
from langchain_tavily import TavilySearch

# Environment setup
def setup_environment():
    """Setup environment variables from .env file or environment"""
    from dotenv import load_dotenv
    import os
    
    # Try to load from .env
    env_files = ['.env']
    for env_file in env_files:
        if os.path.exists(env_file):
            load_dotenv(env_file)
            print(f"[OK] Loaded environment from {env_file}")
            break
    else:
        print("[WARNING] No .env file found, using system environment variables")
    
    # Check for required keys but don't raise error immediately
    # Keys will be provided by user through the frontend interface
    required_keys = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"[WARNING] Missing API keys: {missing_keys}. They will need to be provided through the frontend interface.")
    
    # Optional keys
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "immigrationgptv2"
    
    print("[OK] Environment configured successfully!")

# Vector Store and Document Processing
class ImmigrationRAGSystem:
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = None
        self.retriever = None
        self.setup_vector_store()
    
    def create_vector_store(self, documents, collection_name: str):
        """Create a Qdrant vector store."""
        return Qdrant.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            location=":memory:",
            collection_name=collection_name
        )
    
    def setup_vector_store(self):
        """Initialize the vector store with immigration documents."""
        try:
            # Load documents
            directory_loader = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyMuPDFLoader)
            docs = directory_loader.load()
            
            if not docs:
                print("Warning: No documents found in data directory")
                return
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            split_docs = text_splitter.split_documents(docs)
            
            # Create vector store
            self.vector_store = self.create_vector_store(split_docs, "immigrationgptv2-collection")
            self.retriever = self.vector_store.as_retriever()
            print(f"Vector store created with {len(split_docs)} document chunks")
            
        except Exception as e:
            print(f"Error setting up vector store: {e}")
            self.vector_store = None
            self.retriever = None

# RAG system will be initialized in ImmigrationGPT class


# RAG System Implementation
class ImmigrationRAGAgent:
    def __init__(self, rag_system: ImmigrationRAGSystem):
        self.rag_system = rag_system
        self.generator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.setup_rag_prompt()
    
    def setup_rag_prompt(self):
        """Setup the RAG prompt template."""
        self.human_template = """
#CONTEXT:
{context}

QUERY:
{query}

You are an expert immigration policy assistant specializing in US immigration law and policies. 
Your role is to provide accurate, helpful, and comprehensive information about US immigration.

INSTRUCTIONS:
1. Use the provided context to answer the question accurately and comprehensively
2. If the context contains relevant information, provide a detailed, well-structured response
3. Include specific details, requirements, procedures, or regulations when available
4. If you cannot answer based on the context, politely explain that you need more specific information
5. Always focus on US immigration policies, forms, procedures, and regulations
6. Structure your response clearly with bullet points or numbered lists when appropriate
7. If mentioning forms or documents, include their official names (e.g., Form I-130, Form I-485)

RESPONSE FORMAT:
- Start with a direct answer to the question
- Provide detailed explanations with specific information from the context
- Include relevant requirements, procedures, or regulations
- End with any additional helpful information or next steps if applicable

Remember: You are helping people understand complex immigration processes, so be thorough but clear.
"""
        
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("human", self.human_template)
        ])
    
    def retrieve_and_generate(self, question: str) -> str:
        """Retrieve relevant documents and generate a response."""
        try:
            if not self.rag_system.retriever:
                return "I'm sorry, but I don't have access to the immigration policy database at the moment. Please try again later or ask about current immigration information."
            
            print(f"[SEARCH] Searching knowledge base for: {question[:50]}...")
            
            # Retrieve relevant documents with increased k for better context
            retrieved_docs = self.rag_system.retriever.invoke(question)
            
            if not retrieved_docs:
                return "I couldn't find specific information about this topic in my knowledge base. You might want to ask about current immigration policies or try rephrasing your question."
            
            # Format context with better structure
            context_parts = []
            for i, doc in enumerate(retrieved_docs[:5]):  # Limit to top 5 most relevant docs
                if doc.page_content.strip():
                    context_parts.append(f"Source {i+1}:\n{doc.page_content.strip()}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            print(f"[FOUND] Found {len(retrieved_docs)} relevant documents")
            
            # Generate response
            generator_chain = self.chat_prompt | self.generator_llm | StrOutputParser()
            response = generator_chain.invoke({
                "query": question, 
                "context": context
            })
            
            print(f"[SUCCESS] Generated response from knowledge base")
            return response
            
        except Exception as e:
            print(f"[ERROR] Error in RAG retrieval: {e}")
            return f"I encountered an error while searching my knowledge base: {str(e)}. Please try asking about current immigration information or rephrase your question."






# Agent System Implementation
class ImmigrationAgentSystem:
    def __init__(self, rag_agent: ImmigrationRAGAgent):
        self.rag_agent = rag_agent
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.setup_tools()
        self.setup_agents()
        self.setup_graph()
    
    def setup_tools(self):
        """Setup Tavily search tool."""
        self.tavily_search_tool = TavilySearch(
            max_results=5,
            topic="general",
            include_domains=["uscis.gov", "immigrationforum.org", "migrationpolicy.org"],
            search_depth="advanced"
        )
        
        # Create RAG tool
        @tool
        def retrieve_information(query: str) -> str:
            """Use Retrieval Augmented Generation to retrieve information about immigration policies related to United States of America"""
            return self.rag_agent.retrieve_and_generate(query)
        
        self.rag_tool = retrieve_information
        self.tools = [self.tavily_search_tool, self.rag_tool]
    
    def setup_agents(self):
        """Setup individual agents."""
        # RAG Agent
        self.rag_agent_executor = self.create_agent(
            self.llm,
            [self.rag_tool],
            """You are an expert immigration policy research assistant specializing in US immigration law. 
            Your primary function is to retrieve and provide detailed information about US immigration policies, 
            forms, procedures, and regulations from the knowledge base.
            
            GUIDELINES:
            - Always use the RAG tool first to search for relevant information
            - Provide comprehensive, accurate responses based on the retrieved documents
            - Include specific details like form numbers, requirements, procedures, and timelines
            - Structure your responses clearly with bullet points or numbered lists
            - If the information is not available in the knowledge base, indicate this clearly
            - Focus exclusively on US immigration policies and procedures
            - Be thorough but concise in your explanations"""
        )
        
        # Web Search Agent
        self.web_search_agent_executor = self.create_agent(
            self.llm,
            [self.tavily_search_tool],
            """You are an immigration policy research assistant specializing in current US immigration information. 
            Your role is to search for and provide up-to-date information about US immigration policies, 
            recent changes, and current procedures.
            
            GUIDELINES:
            - Use web search to find the most current information about US immigration policies
            - Focus on official sources like USCIS.gov, immigrationforum.org, and migrationpolicy.org
            - Provide accurate, well-structured summaries of the search results
            - Include relevant dates, policy changes, and current requirements
            - Cross-reference information when possible
            - Focus exclusively on US immigration matters
            - Be thorough in explaining complex immigration processes"""
        )
        
        # Supervisor Agent
        self.supervisor = self.create_team_supervisor(
            self.llm,
            """You are a supervisor managing a team of immigration policy research assistants. 
            Your team consists of:
            - rag-agent: Expert in retrieving information from the immigration policy knowledge base
            - web-search-agent: Expert in finding current immigration information from web sources
            
            DECISION LOGIC:
            1. For questions about established immigration policies, forms, procedures, or regulations: 
               Start with rag-agent to search the knowledge base
            2. For questions about recent changes, current status, or up-to-date information: 
               Use web-search-agent
            3. For complex questions that might need both sources: 
               Start with rag-agent, then web-search-agent if needed
            4. Always prioritize providing comprehensive, accurate information about US immigration
            
            WORKFLOW:
            - Analyze the user's question to determine the best approach
            - Route to the appropriate agent(s) based on the question type
            - Ensure the team provides complete, accurate information
            - When sufficient information has been gathered, respond with FINISH
            
            Remember: Your team only handles US immigration policy research. 
            Do not assign tasks unrelated to immigration policies."""
            ,
            ["rag-agent", "web-search-agent"],
        )
    
    def create_agent(self, llm: ChatOpenAI, tools: list, system_prompt: str) -> AgentExecutor:
        """Create a function-calling agent."""
        system_prompt += ("\nWork autonomously according to your specialty, using the tools available to you."
        " Do not ask for clarification."
        " Your other team members will collaborate with you with their own specialties."
        " You are chosen for a reason!")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools)
        return executor
    
    def create_team_supervisor(self, llm: ChatOpenAI, system_prompt: str, members: List[str]):
        """Create an LLM-based router."""
        options = ["FINISH"] + members
        function_def = {
            "name": "route",
            "description": "Select the next role.",
            "parameters": {
                "title": "routeSchema",
                "type": "object",
                "properties": {
                    "next": {
                        "title": "Next",
                        "anyOf": [{"enum": options}],
                    },
                },
                "required": ["next"],
            },
        }
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]).partial(options=str(options), team_members=", ".join(members))
        
        return (
            prompt
            | llm.bind_functions(functions=[function_def], function_call="route")
            | JsonOutputFunctionsParser()
        )

    def setup_graph(self):
        """Setup the LangGraph workflow."""
        # Define state
        class ResearchTeamState(TypedDict):
            messages: Annotated[List[BaseMessage], operator.add]
            team_members: List[str]
            next: str
        
        # Agent node helper
        def agent_node(state, agent, name):
            result = agent.invoke(state)
            return {"messages": [HumanMessage(content=result["output"], name=name)]}
        
        # Create agent nodes
        rag_node = functools.partial(agent_node, agent=self.rag_agent_executor, name="rag-agent")
        web_search_node = functools.partial(agent_node, agent=self.web_search_agent_executor, name="web-search-agent")
        
        # Create graph
        self.research_graph = StateGraph(ResearchTeamState)
        
        self.research_graph.add_node("rag-agent", rag_node)
        self.research_graph.add_node("web-search-agent", web_search_node)
        self.research_graph.add_node("supervisor", self.supervisor)
        
        self.research_graph.add_edge("rag-agent", "supervisor")
        self.research_graph.add_edge("web-search-agent", "supervisor")
        
        self.research_graph.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {
                "rag-agent": "rag-agent", 
                "web-search-agent": "web-search-agent", 
                "FINISH": END
            },
        )
        
        self.research_graph.set_entry_point("supervisor")
        self.compiled_graph = self.research_graph.compile()
    
    def process_query(self, message: str) -> dict:
        """Process a user query through the agent system."""
        try:
            print(f"[PROCESSING] Processing query: {message[:100]}...")
            
            # Try RAG first for most immigration questions
            rag_response = self.rag_agent.retrieve_and_generate(message)
            
            # If RAG provides a good response, use it
            if rag_response and "couldn't find specific information" not in rag_response.lower() and "not aware of this information" not in rag_response.lower():
                print(f"[SUCCESS] Using RAG response")
                return {
                    "response": rag_response,
                    "tool_used": "RAG",
                    "tool_description": "Retrieval Augmented Generation - Searched through immigration policy documents and knowledge base"
                }
            
            # If RAG doesn't have the information, try web search
            print(f"[FALLBACK] RAG didn't find information, trying web search...")
            try:
                web_search_result = self.tavily_search_tool.invoke(message)
                if web_search_result:
                    # Use LLM to summarize the web search results
                    summary_prompt = f"""
You are an expert immigration policy assistant. Based on the following web search results about US immigration, provide a comprehensive answer to the user's question.

USER QUESTION: {message}

WEB SEARCH RESULTS:
{web_search_result}

Please provide a clear, accurate, and helpful response about US immigration policies. Focus on official information and current policies.
"""
                    web_response = self.llm.invoke(summary_prompt).content
                    print(f"[SUCCESS] Generated response from web search")
                    return {
                        "response": web_response,
                        "tool_used": "Tavily",
                        "tool_description": "Tavily Web Search - Searched current immigration information from official sources like USCIS.gov"
                    }
            except Exception as e:
                print(f"[ERROR] Web search failed: {e}")
            
            # If both fail, return a helpful message
            return {
                "response": "I apologize, but I couldn't find specific information about this topic in my knowledge base or current sources. Please try asking about established immigration policies, forms, or procedures, or rephrase your question.",
                "tool_used": "None",
                "tool_description": "No tool was able to provide relevant information for this query"
            }
            
        except Exception as e:
            print(f"[ERROR] Error processing query: {e}")
            return {
                "response": f"I encountered an error while processing your request: {str(e)}. Please try again with a different question about US immigration policies.",
                "tool_used": "Error",
                "tool_description": "An error occurred while processing the request"
            }
    
    def combine_responses(self, responses: List[str], original_query: str) -> str:
        """Intelligently combine multiple agent responses."""
        try:
            # Use LLM to combine responses
            combine_prompt = f"""
You are an expert immigration policy assistant. You have received multiple responses about US immigration policies for the following question:

QUESTION: {original_query}

RESPONSES:
{chr(10).join([f"Response {i+1}: {resp}" for i, resp in enumerate(responses)])}

Please combine these responses into a single, comprehensive, and well-structured answer that:
1. Directly addresses the original question
2. Eliminates redundancy between responses
3. Presents information in a logical, easy-to-understand format
4. Includes all relevant details from the responses
5. Maintains accuracy and completeness
6. Uses clear formatting with bullet points or numbered lists when appropriate

Provide a single, comprehensive response:
"""
            
            combined_response = self.llm.invoke(combine_prompt).content
            return combined_response
            
        except Exception as e:
            print(f"[ERROR] Error combining responses: {e}")
            # Fallback: return the last response
            return responses[-1] if responses else "I couldn't process your request properly."


# Main ImmigrationGPT Class
class ImmigrationGPT:
    def __init__(self):
        """Initialize the complete ImmigrationGPT system."""
        setup_environment()
        
        # Initialize components only if API keys are available
        self.rag_system = None
        self.rag_agent = None
        self.agent_system = None
        
        # Check if we have the required API keys
        if os.getenv("OPENAI_API_KEY") and os.getenv("TAVILY_API_KEY"):
            self._initialize_components()
            print("ImmigrationGPT system initialized successfully!")
        else:
            print("ImmigrationGPT system initialized but waiting for API keys to be provided.")
    
    def _initialize_components(self):
        """Initialize the AI components with API keys."""
        try:
            self.rag_system = ImmigrationRAGSystem()
            self.rag_agent = ImmigrationRAGAgent(self.rag_system)
            self.agent_system = ImmigrationAgentSystem(self.rag_agent)
            print("AI components initialized successfully!")
        except Exception as e:
            print(f"Error initializing AI components: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if the system is ready to process requests."""
        return self.agent_system is not None
    
    def ask(self, question: str) -> dict:
        """Ask a question to the ImmigrationGPT system."""
        if not self.is_ready():
            return {
                "response": "The system is not ready yet. Please ensure your API keys are properly configured in the settings.",
                "tool_used": "None",
                "tool_description": "System not initialized - API keys required"
            }
        return self.agent_system.process_query(question)


# Example usage
if __name__ == "__main__":
    try:
        immigration_gpt = ImmigrationGPT()
        
        # Test query
        test_question = "Who can petition for derivative refugee or asylee status for their family members?"
        response = immigration_gpt.ask(test_question)
        print(f"\nResponse: {response}")
        
    except Exception as e:
        print(f"Error initializing system: {e}")
        print("Please check your API keys and try again.")