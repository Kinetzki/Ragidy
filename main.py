from rag import Ragidy
import os
from dotenv import load_dotenv
load_dotenv()

ReliaAgent = Ragidy("redis://localhost:6379")
ReliaAgent.ingest_pdf("sample_file.pdf", "relia_ai")


