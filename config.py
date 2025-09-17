import os, logging, json
from dotenv import load_dotenv

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

# Load environment variables from .env file
load_dotenv()
logger.info({"message": "📂 Loaded environment variables"})

# Configuration settings for the RAG application
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error({"message": "❌ Missing GROQ_API_KEY in .env"})
    raise ValueError("Missing GROQ_API_KEY in .env")

PROFILE_PATH = os.getenv("PROFILE_PATH", "BrightSolutionCompanyProfile.pdf")
MODEL_NAME = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Validate resume path
if not os.path.exists(PROFILE_PATH):
    logger.error({"message": f"❌ PDF not found at {PROFILE_PATH}"})
    raise FileNotFoundError(f"PDF not found at {PROFILE_PATH}")
logger.info({"message": f"✅ Validated resume path: {PROFILE_PATH}"})