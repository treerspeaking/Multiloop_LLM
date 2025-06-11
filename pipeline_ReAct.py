import os
from typing import List, Dict, Any, Optional, Tuple, Union
import time
import re
import random
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
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.agents import AgentAction, AgentFinish 
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
# from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
# from langchain_core.exceptions import RateLimitError

from utils.utils import RateLimitException, EnforceReactPatternCallbackHandler, is_rate_limit_error

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # Fallback for environments where GOOGLE_API_KEY might not be immediately available
    # but ensure it's noted.
    print("Warning: GOOGLE_API_KEY environment variable not set. Gemini models will fail if used.")
    # raise ValueError("Please set the GOOGLE_API_KEY environment variable")


class ReACTRAGPipeline:
    """
    A RAG (Retrieval-Augmented Generation) pipeline that uses the ReACT (Reasoning and Acting)
    framework to improve response quality by incorporating reasoning and tool usage.
    """
    
    def __init__(
        self, 
        docs_dir: str = "data", 
        model_name: str = "gemini-2.0-flash", # Updated model name
        embedding_type: str = "gemini",  # Options: "gemini", "fake", "openai"
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 10,  # Number of documents to retrieve (reduced for typical RAG)
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
        
        if not GOOGLE_API_KEY and self.embedding_type == "gemini":
            print("Warning: GOOGLE_API_KEY not set, Gemini embeddings might fail. Consider 'fake' embeddings for testing.")
            # Optionally, force fake embeddings if API key is missing for critical components
            # self.embedding_type = "fake"

        self.embeddings = self._init_embeddings()
        self.llm = self._init_llm()
        self.vector_store = None
        self.retriever = None
        self.agent_executor = None
        
    def clean_agent_response(self, response: str) -> str:
        """
        Cleans up agent responses by removing trailing markdown code block markers
        and other formatting artifacts.
        
        Args:
            response: The raw response string from the agent
        
        Returns:
            A cleaned response string
        """
        # If None or empty response, return empty string
        if not response:
            return ""
            
        # Remove trailing code block markers
        if "\n```" in response:
            response = response.replace("\n```", "")
        
        # Remove "." in response
        if "." in response and response != "Insufficient information.":
            response = response.replace(".", "")
        
        # Remove "Final Answer:" prefix if present
        if "Final Answer:" in response:
            response = re.sub(r'Final Answer:\s*', '', response, flags=re.IGNORECASE).strip()
        
        # Remove any remaining markdown code block markers
        response = response.replace("```", "")
        
        return response.strip()
        
    def _init_embeddings(self) -> Embeddings:
        """Initialize the embedding model based on the specified type"""
        if self.embedding_type == "gemini":
            if not GOOGLE_API_KEY:
                print("GOOGLE_API_KEY not found. Falling back to fake embeddings for Gemini.")
                return FakeEmbeddings(size=768)
            try:
                return GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=GOOGLE_API_KEY,
                    # task_type="RETRIEVAL_DOCUMENT" # Specify task type for potentially better embeddings
                )
            except Exception as e:
                print(f"Error initializing Gemini embeddings: {e}")
                print("Falling back to fake embeddings...")
                return FakeEmbeddings(size=768)
        elif self.embedding_type == "openai":
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                try:
                    return OpenAIEmbeddings(openai_api_key=openai_api_key)
                except Exception as e:
                    print(f"Error initializing OpenAI embeddings: {e}")
                    print("Falling back to fake embeddings...")
                    return FakeEmbeddings(size=1536)
            else:
                print("OpenAI API key not found. Falling back to fake embeddings.")
                return FakeEmbeddings(size=1536)
        else: # "fake"
            print("Using fake embeddings for testing.")
            return FakeEmbeddings(size=768)
    
    def _init_llm(self):
        """Initialize the LLM"""
        if not GOOGLE_API_KEY :
             raise ValueError("GOOGLE_API_KEY must be set to initialize the LLM.")
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1,
            convert_system_message_to_human=True
        )
        
    # Alternative approach using callbacks to enforce pattern
    
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
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        splits = text_splitter.split_documents(documents)
        print(f"Split into {len(splits)} chunks")
        
        # Use FAISS as default
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
    
    
    def setup_react_agent(self):
        """Set up the ReACT agent with tools including the retriever"""
        if not self.retriever:
            print("Retriever not initialized. Loading documents first.")
            # Attempt to load documents if retriever is missing, assuming default path
            self.load_and_process_documents() 
            if not self.retriever: # If still not initialized after trying
                 raise RuntimeError("Failed to initialize retriever. Agent setup cannot proceed.")

        search_tool = Tool(
            name="search_documents", # More descriptive name
            description="Search for information in the document repository. Use this to find relevant text excerpts from available documents based on a query.",
            func=self._search_documents
        )
        
        calculate_tool = Tool(
            name="calculator",
            description="Useful for performing mathematical calculations on numbers. Input should be a valid mathematical expression.",
            func=self._calculate
        )
        
        tools = [search_tool, calculate_tool]
        
        # ReACT prompt template
        # This prompt is inspired by hwchase17/react agent on LangSmith hub
        react_prompt_str = """You are an assistant with reasoning and tool-using capabilities that answers questions based on document content.

You have access to the following tools:

{tools}

To solve a problem, you must follow a specific reasoning process with these exact components:

```
Thought: Try reason, infer and connect information, you MUST CLEARLY describe your reasoning, then describe what information you need and which tool will help.
Action: The specific tool to use (must be one of [{tool_names}])
Action Input: The precise input to provide to the tool
```

After using a tool, you'll see its output:
```
Observation: [tool output will appear here]
```

You MUST follow this pattern for EVERY step:
1. Write a "Thought" explaining your reasoning
2. Choose an "Action" (exact tool name)
3. Provide the "Action Input"
4. Observe the result
5. Start again with a new "Thought"

For "Thought":
- Clearly articulate your reasoning process
- Explain why you're selecting a particular tool
- Analyze results from previous steps
- Determine if you have enough information to answer

For "Action":
- Must be EXACTLY one of the tool names: [{tool_names}]
- Choose "search_documents" to find information in the document repository
- Choose "calculator" to perform mathematical calculations

For "Action Input":
- For search_documents: Provide a clear, focused search query
- For calculator: Provide a valid mathematical expression

Continue this process until you can provide a final answer. When ready to answer:
```
Thought: I now have sufficient information to answer the question
Final Answer: [your concise answer]
```

IMPORTANT INSTRUCTIONS:
1. Your Final Answer MUST be brief and concise with only a few words or the full complete name of the person/organization/company.
2. If there is not enough information to answer confidently, your Final Answer MUST be exactly "Insufficient information."
3. Always follow the Thought → Action → Action Input → Observation pattern until reaching a Final Answer
4. Never skip steps in the reasoning process
5. Never make up information - rely only on tool outputs

Begin!

Question: {input}
{agent_scratchpad}
"""
        prompt_template = PromptTemplate.from_template(react_prompt_str)
        
        # Create custom callback handler to enforce pattern
        pattern_enforcer = EnforceReactPatternCallbackHandler()
        
        # The create_react_agent function will correctly format tools and tool_names for the prompt
        react_agent = create_react_agent(self.llm, tools, prompt_template)
        
        self.agent_executor = AgentExecutor(
            agent=react_agent,
            tools=tools,
            verbose=True, # Prints to console, good for debugging
            handle_parsing_errors=True,
            return_intermediate_steps=True, # Crucial for capturing steps for JSON output,
            callbacks=[pattern_enforcer]
        )
        
        return self.agent_executor
    
    def _search_documents(self, query: str) -> str:
        """Search for documents related to the query"""
        if not self.retriever:
            # This check is more of a safeguard; setup_react_agent should ensure retriever exists
            print("Error: Document retriever not initialized. Please load documents first.")
            self.load_and_process_documents() # Attempt to initialize
            if not self.retriever:
                return "Error: Document retriever could not be initialized."
            
        docs = self.retriever.invoke(query)
        results = []
        max_length = 300
        
        if not docs:
            return "No relevant documents found for your query."

        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            content_preview = doc.page_content[:max_length] + "..." if len(doc.page_content) > max_length else doc.page_content
            results.append(f"Document {i+1} (Source: {source}):\n{content_preview}\n")
            
        return "\n".join(results)
    
    def _calculate(self, expression: str) -> str:
        """Safely evaluate a mathematical expression"""
        try:
            # Using a safer eval alternative is recommended for production.
            # For this example, simple eval is used.
            # Allow basic math functions from numpy for more power if needed, but keep it simple for now.
            allowed_globals = {"__builtins__": {}} # No builtins for safety
            # For more complex math, consider ast.literal_eval or a dedicated math parsing library like numexpr
            result = eval(expression, allowed_globals, {}) 
            return f"Result: {result}"
        except Exception as e:
            return f"Error calculating '{expression}': {str(e)}. Ensure the expression is purely mathematical (e.g., '2 + 2 * 5')."

    def _serialize_agent_step(self, step: Tuple[Union[AgentAction, AgentFinish], Any]) -> Dict[str, Any]:
        """Converts a single agent step (action/finish and observation) to a serializable dictionary."""
        action_or_finish, observation = step
        
        # Check if this is an AgentFinish object (final answer)
        if hasattr(action_or_finish, 'return_values'):
            # This is an AgentFinish object
            return {
                "finish": {
                    "output": action_or_finish.return_values.get("output", ""),
                    "log": action_or_finish.log.strip() if hasattr(action_or_finish, "log") else ""
                },
                "observation": str(observation) if not isinstance(observation, (str, int, float, list, dict, bool)) else observation
            }
        else:
            # This is an AgentAction object
            action_dict = {
                "tool": action_or_finish.tool,
                "tool_input": action_or_finish.tool_input,
                "log": action_or_finish.log.strip() # The thought process and action invocation
            }
            # Ensure observation is serializable
            serialized_observation = str(observation) if not isinstance(observation, (str, int, float, list, dict, bool)) else observation
            return {"action": action_dict, "observation": serialized_observation}
    
    def query(self, question: str, use_react: bool = True, max_retries: int = 5) -> Dict[str, Any]:
        """
        Query the system with a question.
        
        Args:
            question: The user's question.
            use_react: Whether to use ReACT agent (True) or basic RAG (False).
            max_retries: Maximum number of retries for rate limit errors.
                
        Returns:
            A dictionary containing the query, answer, and inference steps.
        """
        if not GOOGLE_API_KEY:
            # If API key was not set at start, critical operations might fail.
            # Give a more direct error here if trying to query.
            if self.embedding_type == "gemini" or self.model_name.startswith("gemini"):
                raise ValueError(
                    "GOOGLE_API_KEY is not set. Cannot proceed with query using Gemini models or embeddings. "
                    "Please set the GOOGLE_API_KEY environment variable or choose 'fake' embeddings for testing."
                )

        output_data: Dict[str, Any] = {
            "query": question,
            "answer": "Error: Could not generate answer.", # Default error answer
            "inference_steps": [] 
        }

        if use_react:
            if not self.agent_executor:
                print("ReACT agent not set up. Setting it up now...")
                self.setup_react_agent()
            
            if not self.agent_executor: # If setup failed
                output_data["answer"] = "Error: ReACT Agent could not be initialized."
                return output_data

            # Initialize retry count
            retry_count = 0
            while retry_count <= max_retries:
                try:
                    # The response from invoke with return_intermediate_steps=True
                    # will be a dictionary containing 'input', 'output', and 'intermediate_steps'.
                    response = self.agent_executor.invoke({"input": question})
                    
                    raw_answer = response.get("output", "No answer found.")
                    output_data["answer"] = self.clean_agent_response(raw_answer)
                    
                    # Serialize intermediate steps
                    raw_steps = response.get("intermediate_steps", [])
                    output_data["inference_steps"] = [self._serialize_agent_step(step) for step in raw_steps]
                    
                    # If successful, break out of the retry loop
                    break

                except Exception as e:
                    error_str = str(e)
                    print(f"Error during ReACT agent execution: {e}")
                    
                    # Check if this is a rate limit error
                    if is_rate_limit_error(e):
                        retry_count += 1
                        
                        # Try to extract retry delay from error message, default to 30 seconds
                        retry_exception = RateLimitException.from_error(e)
                        retry_after = retry_exception.retry_after or 30
                        
                        # Add jitter to avoid all clients retrying at exactly the same time
                        jitter = random.uniform(0, 5)
                        wait_time = retry_after + jitter
                        
                        if retry_count <= max_retries:
                            print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds (attempt {retry_count}/{max_retries})...")
                            time.sleep(wait_time)
                            continue
                        else:
                            output_data["answer"] = f"Error: Maximum retries exceeded due to rate limits. Please try again later."
                            output_data["inference_steps"] = [{"error": error_str, "retries": retry_count}]
                    else:
                        # For non-rate-limit errors, don't retry
                        output_data["answer"] = f"Error during agent execution: {error_str}"
                        output_data["inference_steps"] = [{"error": error_str}]
                        break

        return output_data

    def save_output_to_json(self, output_data: Dict[str, Any], filename: str = "rag_output.json", append: bool = True):
        """
        Saves the structured output data to a JSON file. Can append to existing file.

        Args:
            output_data: The dictionary containing query, answer, and inference_steps.
            filename: The name of the JSON file to save the output.
            append: If True, appends to existing file; if False, creates a new file.
        """
        try:
            all_outputs = []
            
            # Try to read existing file if we're appending
            if append and os.path.exists(filename):
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        all_outputs = json.load(f)
                    # Make sure it's a list
                    if not isinstance(all_outputs, list):
                        all_outputs = [all_outputs]
                except json.JSONDecodeError:
                    # If the file exists but is empty or malformed, start with an empty list
                    all_outputs = []
            
            # Add the new output
            all_outputs.append(output_data)
            
            # Write all outputs back to the file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(all_outputs, f, indent=4, ensure_ascii=False)
            print(f"Output successfully {'appended to' if append else 'saved to'} {filename}")
        except Exception as e:
            print(f"Error saving output to JSON: {e}")


# Example usage
if __name__ == "__main__":
    
    output_file = "/home/treerspeaking/src/python/Multiloop_LLM/react_rag_output.json"
    ques_file = "/home/treerspeaking/src/python/Multiloop_LLM/input_test.json"
    
    with open(output_file, "w") as json_file:
        json.dump([], json_file, indent=4)
    
    with open(ques_file, 'r', encoding='utf-8') as f:
        json_input = json.load(f)
        
    for item in json_input:
        pipeline = ReACTRAGPipeline(
            docs_dir="data",
            embedding_type="gemini", 
            vector_store_type="faiss"
        )
        
        # Load documents (or use existing index)
        # Set refresh=True if you changed documents or want to rebuild the index
        pipeline.load_and_process_documents(refresh=False) 

        question = item["query"]
        
        print(f"\nQuerying with: '{question}' using ReACT agent...")
        rag_output = pipeline.query(question, use_react=True)
        
        # Print the final answer
        print("\nFormatted Output:")
        print(f"Query: {rag_output['query']}")
        print(f"Answer: {rag_output['answer']}")
        
        # Save the output to JSON
        pipeline.save_output_to_json(rag_output, output_file)
        
