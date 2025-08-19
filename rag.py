from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_redis import RedisVectorStore, RedisChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


class Ragidy:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
    
    def ingest_pdf(self, 
        file_path: str, 
        vector_key: str, 
        chunk_size: int = 2500, 
        char_overlap: int = 250
    ) -> None:
        # load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=char_overlap)
        
        docs = splitter.split_documents(documents)
        
        vector_key = f"asst_{vector_key}"
        embedding = OpenAIEmbeddings()
        
        # store vectors to redis
        RedisVectorStore.from_documents(
            documents=docs,
            embedding=embedding,
            index_name=vector_key,
            redis_url=self.redis_url
        )


    def create_history_retriever(self, 
        vector_store: RedisVectorStore,
        llm: str = "gpt-5"
    ):
        rewriter_llm = ChatOpenAI(llm=llm, temperature=0.0)
        retriever = vector_store.as_retriever(
            search_type= "similarity",
            search_kwargs={"k": 5})
        