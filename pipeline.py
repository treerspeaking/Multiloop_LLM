"""
ReACT RAG Pipeline using LangChain and Google's Gemini API
This script implements a Retrieval-Augmented Generation system with ReACT (Reasoning and Acting)
capabilities using LangChain and the Gemini API.

Required packages:
pip install langchain langchain-google-genai langchain-community faiss-cpu google-generativeai
"""

import os
from typing import List, Dict, Any, Optional
import json
import numpy as np
from pydantic import BaseModel, Field

# LangChain imports
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import FakeEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import AgentExecutor, create_react_agent, initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable")

class ReACTRAGPipeline:
    """
    A RAG (Retrieval-Augmented Generation) pipeline that uses the ReACT (Reasoning and Acting)
    framework to improve response quality by incorporating reasoning and tool usage.
    """
    
    def __init__(
        self, 
        docs_dir: str = "data", 
        model_name: str = "gemini-2.5-flash-preview-04-17", 
        embedding_type: str = "gemini",  # Options: "gemini", "fake", "openai"
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 50,  # Number of documents to retrieve
        vector_store_type: str = "faiss"  # Options: "faiss", "chroma"
    ):
        """
        Initialize the ReACT RAG pipeline.
        
        Args:
            docs_dir: Directory containing documents to index
            model_name: Gemini model to use
            embedding_type: Type of embeddings to use ("gemini", "fake", or "openai")
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            k: Number of documents to retrieve
            vector_store_type: Type of vector store to use ("faiss" or "chroma")
        """
        self.docs_dir = docs_dir
        self.model_name = model_name
        self.embedding_type = embedding_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.vector_store_type = vector_store_type
        
        # Initialize components
        self.embeddings = self._init_embeddings()
        self.llm = self._init_llm()
        self.vector_store = None
        self.retriever = None
        self.agent_executor = None
        
    def _init_embeddings(self) -> Embeddings:
        """Initialize the embedding model based on the specified type"""
        if self.embedding_type == "gemini":
            # Use Google's Gemini embeddings
            try:
                return GoogleGenerativeAIEmbeddings(
                    # change to better embedding network
                    task_type=None,
                    model="models/text-embedding-004",
                    google_api_key=GOOGLE_API_KEY
                )
            except Exception as e:
                print(f"Error initializing Gemini embeddings: {e}")
                print("Falling back to fake embeddings...")
                return FakeEmbeddings(size=768)
        elif self.embedding_type == "openai":
            # Use OpenAI embeddings if API key is available
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                try:
                    return OpenAIEmbeddings()
                except Exception as e:
                    print(f"Error initializing OpenAI embeddings: {e}")
                    print("Falling back to fake embeddings...")
                    return FakeEmbeddings(size=1536)
            else:
                print("OpenAI API key not found. Falling back to fake embeddings.")
                return FakeEmbeddings(size=1536)
        else:
            # Use fake embeddings as a fallback
            print("Using fake embeddings for testing.")
            return FakeEmbeddings(size=768)
    
    def _init_llm(self):
        """Initialize the LLM"""
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            convert_system_message_to_human=True
        )
    
    def load_and_process_documents(self, docs_dir: Optional[str] = None, refresh=False):
        """
        Load and process documents from a directory.
        
        Args:
            docs_dir: Directory to load documents from. If None, uses self.docs_dir
        """
        save_path = "faiss_index"
        
        if not refresh and os.path.exists(save_path):
            print("Loading existing vector store")
            self.vector_store = FAISS.load_local(save_path, self.embeddings, allow_dangerous_deserialization=True)
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.k}
            )
            return 
        

        if docs_dir:
            self.docs_dir = docs_dir
            
        # Load documents
        try:
            loader = DirectoryLoader(
                self.docs_dir, 
                glob="**/*.txt", 
                loader_cls=TextLoader
            )
            documents = loader.load()
            print(f"Loaded {len(documents)} documents from {self.docs_dir}")
        except Exception as e:
            print(f"Error loading documents: {e}")
            # Create sample documents if loading fails
            print("Creating sample documents instead...")
            documents = self._create_sample_documents()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        splits = text_splitter.split_documents(documents)
        print(f"Split into {len(splits)} chunks")
        
        # Create vector store
        if self.vector_store_type == "chroma":
            try:
                self.vector_store = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings
                )
            except Exception as e:
                print(f"Error creating Chroma vector store: {e}")
                print("Falling back to FAISS...")
                self.vector_store_type = "faiss"
                self.vector_store = None
        
        # Use FAISS as default or fallback
        if self.vector_store_type == "faiss" or self.vector_store is None:
            try:
                self.vector_store = FAISS.from_documents(
                    documents=splits,
                    embedding=self.embeddings
                )
                self.vector_store.save_local(save_path)
                print(f"Vector store saved to {save_path}")
                
            except Exception as e:
                print(f"Error creating FAISS vector store: {e}")
                raise RuntimeError("Failed to create any vector store.")
        
        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k}
        )
        
        # return splits
    
    def _create_sample_documents(self) -> List[Document]:
        """Create sample documents for testing"""
        sample_texts = [
            "LangChain is a framework for developing applications powered by language models.",
            "RAG stands for Retrieval-Augmented Generation, which enhances LLM outputs with external knowledge.",
            "ReACT is a framework that combines reasoning and acting for improved decision making in AI systems.",
            "Gemini is a family of multimodal large language models developed by Google.",
            "Vector databases store embeddings of documents for semantic search capabilities."
        ]
        
        return [Document(page_content=text, metadata={"source": f"sample_{i}.txt"}) 
                for i, text in enumerate(sample_texts)]
    
    def setup_react_agent(self):
        """Set up the ReACT agent with tools including the retriever"""
        # Define the search tool
        search_tool = Tool(
            name="search",
            description="Search for information in the document repository",
            func=self._search_documents
        )
        
        # Define the calculate tool
        calculate_tool = Tool(
            name="calculator",
            description="Useful for performing calculations",
            func=self._calculate
        )
        
        tools = [search_tool, calculate_tool]
        
        # Define the ReACT agent
        react_prompt = """You are an assistant with reasoning and tool-using capabilities. 
You have access to the following tools:

{tools}

To use a tool, please use the following format:
```
Thought: I need to think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
```

After you use a tool, I'll show you the output:
```
Observation: tool output
```

Based on the observation, you should continue with "Thought/Action/Action Input" until you can provide a final answer.
When you're ready to give a final answer, use:
```
Thought: I can now answer the question
Final Answer: [your answer here]
```

Begin!

Question: {input}
{agent_scratchpad}
"""

        prompt = PromptTemplate(
            template=react_prompt,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={"tools": tools, "tool_names": ", ".join([tool.name for tool in tools])}
        )
        
        react_agent = create_react_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=react_agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        return self.agent_executor
    
    def _search_documents(self, query: str) -> str:
        """Search for documents related to the query"""
        if not self.retriever:
            return "Error: Document retriever not initialized. Please load documents first."
            
        docs = self.retriever.invoke(query)
        results = []
        
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            content = doc.page_content
            results.append(f"Document {i+1} (Source: {source}):\n{content}\n")
            
        return "\n".join(results) if results else "No relevant documents found."
    
    def _calculate(self, expression: str) -> str:
        """Safely evaluate a mathematical expression"""
        try:
            # Using eval for simplicity - in production use a safer alternative like numexpr
            result = eval(expression, {"__builtins__": {}})
            return f"Result: {result}"
        except Exception as e:
            return f"Error calculating '{expression}': {str(e)}"
    
    # def setup_rag_chain(self):
    #     """Set up a basic RAG chain"""
    #     if not self.retriever:
    #         raise ValueError("Retriever not initialized. Please load documents first.")
            
    #     # Define the RAG prompt
    #     rag_prompt = PromptTemplate.from_template("""
    #     You are a helpful AI assistant. Use the following retrieved documents to answer the question.
    #     If you don't know the answer, just say you don't know. Don't try to make up an answer.
        
    #     Retrieved documents:
    #     {context}
        
    #     Question: {question}
        
    #     Answer:
    #     """)
        
    #     # Create the RAG chain
    #     rag_chain = (
    #         {"context": self.retriever, "question": RunnablePassthrough()}
    #         | rag_prompt
    #         | self.llm
    #         | StrOutputParser()
    #     )
        
    #     return rag_chain
    
    def query(self, question: str, use_react: bool = True) -> str:
        """
        Query the system with a question
        
        Args:
            question: The user's question
            use_react: Whether to use ReACT agent (True) or basic RAG (False)
            
        Returns:
            The system's response
        """
        if use_react:
            if not self.agent_executor:
                self.setup_react_agent()
            return self.agent_executor.invoke({"input": question})
        # else:
        #     rag_chain = self.setup_rag_chain()
        #     return rag_chain.invoke(question)

# Example usage
if __name__ == "__main__":
    # Initialize the pipeline
    # You can customize the embedding type and vector store type based on your available dependencies
    # Options for embedding_type: "gemini" (default, requires Gemini API key), "fake" (for testing), "openai" (requires OpenAI API key)
    # Options for vector_store_type: "faiss" (default), "chroma" (may require additional dependencies)
    pipeline = ReACTRAGPipeline(
        embedding_type="gemini",  # Use "fake" if you don't have API keys
        vector_store_type="faiss"  # FAISS has fewer dependencies than Chroma
    )
    
    # Load documents
    pipeline.load_and_process_documents()
    
    # Set up the ReACT agent
    pipeline.setup_react_agent()
    
    # # Example query using ReACT
    question = "Do the TechCrunch article on software companies and the Hacker News article on The Epoch Times both report an increase in revenue related to payment and subscription models, respectively?"
    result = pipeline.query(question, use_react=True)
    print("\nReACT Answer:")
    if isinstance(result, dict) and "output" in result:
        print(result["output"])
    else:
        print(result)
    
    # # Example query using basic RAG
    # result = pipeline.query(question, use_react=False)
    # print("\nBasic RAG Answer:")
    # print(result)