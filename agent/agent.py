import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import re
import math
from chromadb import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools.python.tool import PythonAstREPLTool

load_dotenv()

class DictionaryTool(BaseTool):
    name = "dictionary_lookup"
    description = "Look up the definition of a word or term"
    
    def _run(self, word: str) -> str:
        # Simplified dictionary with a few sample definitions
        dictionary = {
            "rag": "Retrieval Augmented Generation, a technique that combines LLMs with information retrieval",
            "llm": "Large Language Model, an AI model trained on large text corpora to generate human-like text",
            "vector": "In machine learning, a numerical representation of data in an n-dimensional space",
            "embedding": "A dense vector representation of text or other data that captures semantic meaning",
            "agent": "In AI, a system that perceives its environment and takes actions to achieve goals",
            "faiss": "Facebook AI Similarity Search, a library for efficient similarity search and clustering of dense vectors",
        }
        
        # Case-insensitive lookup
        word = word.lower()
        if word in dictionary:
            return dictionary[word]
        else:
            return f"Definition for '{word}' not found in the dictionary."

class RetrievalTool(BaseTool):
    name = "document_retrieval"
    description = "Retrieve information from the document collection based on the query"
    
    def __init__(self, vectorstore_path: str = "faiss_index"):
        super().__init__()
        # Load the vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2", 
            model_kwargs={"device": "cpu"}
        )
        # self.vectorstore = FAISS.load_local(vectorstore_path, embeddings)
        # MODIFIED LINE:
        
        vector_store = Chroma(
        collection_name="rag_collection", # Same name used during ingestion
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY
        )
    def _run(self, query: str) -> str:
        # Retrieve the top 3 most relevant document chunks
        docs = self.vectorstore.similarity_search(query, k=3)
        
        # Format the results with source information
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            page_info = f" (Page {page})" if page else ""
            
            results.append(f"[Document {i}] From {source}{page_info}:\n{doc.page_content}\n")
        
        return "\n".join(results)

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Perform mathematical calculations"
    
    def _run(self, expression: str) -> str:
        # Using PythonAstREPLTool for safe evaluation
        calculator = PythonAstREPLTool()
        
        # Clean the expression
        expression = expression.replace("^", "**")
        
        try:
            result = calculator._run(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error calculating '{expression}': {str(e)}"

def initialize_agent():
    """Initialize the agent with tools and LLM."""
    # Set up the LLM
    llm = ChatGroq(
        temperature=0.7,
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama2-70b-4096"
    )
    
    # Initialize tools
    tools = [
        RetrievalTool(),
        DictionaryTool(),
        CalculatorTool()
    ]
    
    # Create a custom prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a helpful assistant with access to the following tools:
1. document_retrieval: Use this to find information in the document collection
2. dictionary_lookup: Use this to look up definitions of terms
3. calculator: Use this to perform mathematical calculations

For each user query:
- If the query is about a calculation or involves math, use the calculator tool
- If the query is asking for a definition, use the dictionary tool
- For all other queries, use the document_retrieval tool to find relevant information

Always explain your reasoning process and cite the sources of information when using document retrieval.
Your final answer should be comprehensive and helpful."""),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    
    return agent_executor

def process_query(agent_executor, query: str, chat_history: List = None):
    """Process a user query through the agent."""
    if chat_history is None:
        chat_history = []
    
    # Log the query type for demonstration purposes
    query_type = "unknown"
    if re.search(r'calculate|compute|math|sum|multiply|divide|subtract|add|\+|\-|\*|\/|\^', query, re.IGNORECASE):
        query_type = "calculation"
    elif re.search(r'define|meaning|definition|what is a|what is the|what are', query, re.IGNORECASE):
        query_type = "definition"
    else:
        query_type = "document_retrieval"
    
    # Execute the agent
    result = agent_executor.invoke({
        "input": query,
        "chat_history": chat_history
    })
    
    # Extract the tool used from the agent's thoughts
    tools_used = []
    if "intermediate_steps" in result:
        for step in result["intermediate_steps"]:
            if hasattr(step[0], "tool") and step[0].tool:
                tools_used.append(step[0].tool)
    
    return {
        "query_type": query_type,
        "answer": result["output"],
        "tools_used": tools_used,
        "full_result": result
    }

if __name__ == "__main__":
    # Simple CLI for testing
    agent = initialize_agent()
    
    print("RAG-Powered Multi-Agent Q&A System")
    print("Type 'exit' to quit")
    
    chat_history = []
    
    while True:
        query = input("\nEnter your question: ")
        
        if query.lower() == 'exit':
            break
        
        result = process_query(agent, query, chat_history)
        
        print(f"\nQuery Type: {result['query_type']}")
        if result['tools_used']:
            print(f"Tools Used: {', '.join(result['tools_used'])}")
        print("\nAnswer:")
        print(result['answer'])
        
        # Update chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result['answer']))
        print("\n" + "-" * 50)
        print("\n") 