from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import httpx, os, logging, cachetools, json, warnings
from config import GROQ_API_KEY, PROFILE_PATH, MODEL_NAME, EMBEDDING_MODEL 

# Ignore warnings
warnings.filterwarnings("ignore")

# Configure console-only JSON logging with emojis
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Custom JSON log formatter for structured output
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage()
        }
        return json.dumps(log_entry)

# Apply JSON formatter to console handler
for handler in logging.getLogger().handlers:
    handler.setFormatter(JSONFormatter())

class CustomDocChatbot:
    """A RAG-based chatbot for answering questions using a resume PDF."""
    
    # In-memory cache for query responses
    query_cache = cachetools.TTLCache(maxsize=100, ttl=300)  # Cache for 5 minutes

    def __init__(self):
        """Initialize the chatbot with LLM, embeddings, and RAG chain."""
        self.llm = self.configure_llm()
        self.embeddings = self.configure_embedding_model()
        self.vector_db = None  # Initialize later in setup_qa_chain
        self.qa_chain = self.setup_qa_chain()
        self.http_client = httpx.AsyncClient(timeout=15.0)
        logger.info({"message": "🤖 CustomDocChatbot initialized"})

    def configure_llm(self):
        """Configure the Groq LLM with specified model and API key.
        
        Returns:
            ChatGroq: Configured LLM instance.
        
        Raises:
            Exception: If LLM configuration fails.
        """
        try:
            llm = ChatGroq(
                model_name=MODEL_NAME,
                temperature=0.5,
                groq_api_key=GROQ_API_KEY
            )
            logger.info({"message": "✅ Groq LLM configured successfully"})
            return llm
        except Exception as e:
            logger.error({"message": f"❌ Failed to configure LLM: {str(e)}"})
            raise

    def configure_embedding_model(self):
        """Configure HuggingFace embeddings for CPU compatibility.
        
        Returns:
            HuggingFaceEmbeddings: Configured embedding model.
        
        Raises:
            Exception: If embedding configuration fails.
        """
        try:
            os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_transformers"
            os.environ["HF_HOME"] = "/tmp/huggingface"
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info({"message": "✅ HuggingFace embeddings configured"})
            return embeddings
        except Exception as e:
            logger.error({"message": f"❌ Failed to configure embeddings: {str(e)}"})
            raise

    def load_pdf(self):
        """Load and validate the resume PDF document.
        
        Returns:
            list: List of document pages.
        
        Raises:
            FileNotFoundError: If the PDF file is not found.
            Exception: If PDF loading fails.
        """
        try:
            if not os.path.exists(PROFILE_PATH):
                raise FileNotFoundError(f"PDF not found at {PROFILE_PATH}")
            loader = PyPDFLoader(PROFILE_PATH)
            docs = loader.load()
            logger.info({"message": f"📄 Loaded {len(docs)} pages from {PROFILE_PATH}"})
            return docs
        except Exception as e:
            logger.error({"message": f"❌ Error loading PDF: {str(e)}"})
            raise

    def setup_qa_chain(self):
        """Configure the RAG chain with in-memory FAISS vector store and memory.
        
        Returns:
            ConversationalRetrievalChain: Configured QA chain.
        
        Raises:
            Exception: If chain setup fails.
        """
        try:
            # Load and split PDF into chunks
            docs = self.load_pdf()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600, chunk_overlap=200, add_start_index=True
            )
            splits = text_splitter.split_documents(docs)
            logger.info({"message": f"📑 Created {len(splits)} chunks"})

            # Initialize in-memory FAISS vector store (no disk persistence)
            self.vector_db = FAISS.from_documents(splits, self.embeddings)
            logger.info({"message": "🔍 FAISS vector store initialized in-memory"})

            # Initialize retriever with MMR for diverse results
            retriever = self.vector_db.as_retriever(
                search_type="mmr", search_kwargs={"k": 4, "fetch_k": 5}
            )
            logger.info({"message": "🔍 FAISS retriever initialized"})

            # Set up conversation memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                output_key="answer",
                return_messages=True,
                memory_limit=10
            )

            # Define the prompt template (unchanged as requested)
            prompt_template = """
                You are Bright Solutions Assistant — a confident, calm, and friendly BPO company representative 🤖✨.  
                Your goal is to provide clear, accurate, and helpful communication at all times.  
                
                ✅ Always analyze the **question** and reply accordingly based on the context.  
                    - If the question is relevant to the company, services, or context → answer directly, clearly, and concisely.  
                    - If the question is **irrelevant** (not related to the company or services) → politely respond with: 
                        "I cannot answer that right now as it’s not related to Bright Solutions. 😊  
                        Please feel free to reach us at Phone: +1 (832) 390-6434 or +92 333-316-7749,  
                        or Email: info@brightssolution.com, and our team will gladly assist you."  
                
                💬 Maintain a professional, friendly, and approachable tone — always sound calm and confident.  
                ✨ Use relevant emojis to make responses engaging, but never overuse them.  
                
                Context:  
                {context}  
                
                Question:  
                {question}  
                
                Answer:
                """

            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
            
            # Initialize the RAG chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt},
                return_source_documents=False,
                verbose=False
            )
            logger.info({"message": "🚀 QA chain initialized"})
            return qa_chain
        except Exception as e:
            logger.error({"message": f"❌ Error setting up QA chain: {str(e)}"})
            raise

    async def query(self, question: str) -> str:
        """Process a user query through the RAG chain with caching.
        
        Args:
            question (str): User's question to answer.
        
        Returns:
            str: Processed response with <think> tags removed.
        
        Raises:
            Exception: If query processing fails.
        """
        try:
            # Check cache first
            if question in self.query_cache:
                response = self.query_cache[question]
                logger.info({"message": f"💾 Cache hit for query: {question}"})
                return response

            # Process query through RAG chain
            result = await self.qa_chain.ainvoke({"question": question})
            response = result["answer"].strip()
            self.query_cache[question] = response  # Cache the response
            logger.info({"message": f"💬 Query: {question} | Answer: {response}"})
            return response
        except Exception as e:
            logger.error({"message": f"❌ Query error: {str(e)}"})
            raise

    async def shutdown(self):
        """Clean up resources on shutdown."""
        await self.http_client.aclose()
        logger.info({"message": "🛑 HTTP client closed"})