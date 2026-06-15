import os
import dotenv

dotenv.load_dotenv()

print(os.getenv("GEMINI_API_KEY"))