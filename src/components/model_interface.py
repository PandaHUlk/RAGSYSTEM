import os
import sys
from typing import Dict, List, Optional, Tuple, Union

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

from src.exception import CustomException
from src.logger import logging


class ModelInterface:
    """
    Component to interact with language models
    """
    def __init__(
        self,
        model_provider: str = "ollama",
        model_name: str = "llama2",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: Optional[str] = None
    ):
        """
        Initialize model interface
        
        Args:
            model_provider: Provider of the language model ('openai' or 'ollama')
            model_name: Name of the model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            api_key: API key for model provider (if needed)
        """
        self.model_provider = model_provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize LLM
        try:
            if model_provider == "gemini-ai":
                if not ChatGoogleGenerativeAI:
                    raise ImportError("langchain_openai not installed. Install with pip install langchain-openai")
                
                # Use API key from parameter or environment variable
                self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
                if not self.api_key:
                    raise ValueError("GOOGLEAI API key is required")
                
                self.llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=self.api_key
                )
                logging.info(f"Initialized OpenAI model: {model_name}")
                
            elif model_provider == "ollama":
                # For local Ollama models
                self.llm = Ollama(
                    model=model_name,
                    temperature=temperature,
                    num_ctx=max_tokens
                )
                logging.info(f"Initialized Ollama model: {model_name}")
                
            else:
                raise ValueError(f"Unknown model provider: {model_provider}")
                
        except Exception as e:
            logging.error(f"Error initializing language model: {e}")
            raise CustomException(e, sys)
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def _format_docs(self, docs: List[Dict]) -> str:
        """
        Format documents for context
        """
        formatted_docs = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            topic = doc.metadata.get("topic", "Unclassified")
            formatted_docs.append(f"[Document {i+1}] Source: {source} | Topic: {topic}\n{doc.page_content}")
        
        return "\n\n".join(formatted_docs)
    
    def setup_rag_chain(self) -> None:
        """
        Set up the RAG chain for question answering
        """
        try:
            # Define prompt templates
            condense_question_template = """Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that includes all necessary context from the conversation.

            Chat History:
            {chat_history}
            
            Follow-up question: {question}
            
            Standalone question:"""
            
            condense_question_prompt = PromptTemplate.from_template(condense_question_template)
            
            qa_template = """You are a helpful AI assistant that answers questions based on the provided documents. 
            
            Use ONLY the following context to answer the question. If you don't know the answer based on the context, say you don't know, but try to be helpful and suggest what might be relevant.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
            
            qa_prompt = PromptTemplate.from_template(qa_template)
            
            # Create question condenser chain
            self.condense_question_chain = (
                {"question": RunnablePassthrough(), "chat_history": lambda _: self.memory.load_memory_variables({})["chat_history"]}
                | condense_question_prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Create QA chain
            self.qa_chain = (
                {"context": lambda x: self._format_docs(x["context"]), 
                 "question": lambda x: x["question"]}
                | qa_prompt
                | self.llm
                | StrOutputParser()
            )
            
            logging.info("RAG chain setup complete")
            
        except Exception as e:
            logging.error(f"Error setting up RAG chain: {e}")
            raise CustomException(e, sys)
    
    def answer_question(
        self,
        question: str,
        docs: List[Dict],
        chat_mode: bool = True
    ) -> str:
        """
        Answer a question using the provided documents
        
        Args:
            question: User question
            docs: Relevant documents from vector store
            chat_mode: Whether to use chat history
            
        Returns:
            Model's answer
        """
        try:
            if not hasattr(self, "qa_chain"):
                self.setup_rag_chain()
            
            if chat_mode and len(self.memory.load_memory_variables({})["chat_history"]) > 0:
                # Use conversation history to get better standalone question
                standalone_question = self.condense_question_chain.invoke(question)
                logging.info(f"Condensed question: {standalone_question}")
            else:
                standalone_question = question
            
            # Get answer using RAG chain
            answer = self.qa_chain.invoke({"context": docs, "question": standalone_question})
            
            # Update memory
            if chat_mode:
                self.memory.save_context(
                    {"input": question},
                    {"output": answer}
                )
            
            return answer
            
        except Exception as e:
            logging.error(f"Error answering question: {e}")
            raise CustomException(e, sys)
    
    def clear_chat_history(self) -> None:
        """
        Clear conversation memory
        """
        try:
            self.memory.clear()
            logging.info("Chat history cleared")
        except Exception as e:
            logging.error(f"Error clearing chat history: {e}")
            raise CustomException(e, sys)