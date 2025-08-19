from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_redis import RedisVectorStore, RedisChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks import AsyncIteratorCallbackHandler

class StreamHandler(AsyncIteratorCallbackHandler):
    def __init__(self):
        self.answer_started = False
        self.done = False
        self.title = []
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        pass
    
    def on_llm_end(self, response, **kwargs):
        self.answer_started = True
    
    def on_llm_new_token(self, token, **kwargs):
        if self.answer_started:
            print(token)
        else:
            self.title.append(token)
    

class Ragidy:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
    
    async def ingest_pdf(self, 
        file_path: str, 
        vector_key: str, 
        chunk_size: int = 2500, 
        char_overlap: int = 250
    ) -> None:
        # load PDF
        loader = PyPDFLoader(file_path)
        documents = await loader.aload()
        
        # split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=char_overlap)
        
        docs = splitter.split_documents(documents)
        
        vector_key = f"asst_{vector_key}"
        embedding = OpenAIEmbeddings()
        
        # store vectors to redis
        await RedisVectorStore.afrom_documents(
            documents=docs,
            embedding=embedding,
            index_name=vector_key,
            redis_url=self.redis_url
        )

    def get_pdf_vector_store(self, index_name: str) -> RedisVectorStore:
        embedding = OpenAIEmbeddings()
        vector_key = f"asst_{index_name}"
        vector_store = RedisVectorStore.from_existing_index(
            index_name=vector_key,
            embedding=embedding,
            redis_url=self.redis_url
        )

        return vector_store
    
    def create_history_retriever(self,
        vector_store: RedisVectorStore,
        llm: str = "gpt-5"
    ):
        rewriter_llm = ChatOpenAI(model=llm, temperature=0.0)
        retriever = vector_store.as_retriever(
            search_type= "similarity",
            search_kwargs={"k": 5})
        
        prompt_search_query = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "system",
                "Produce a short search query or keywords to find documents relevant to answering the user's question."
            )
        ])
        
        history_retriever = create_history_aware_retriever(
            llm=rewriter_llm,
            retriever=retriever,
            prompt=prompt_search_query
        )
        
        return history_retriever
    
    def create_doc_chain(self, llm: str = "gpt-5"):
        doc_llm = ChatOpenAI(model=llm, temperature=0, max_completion_tokens=1000)
        
        prompt_answer = ChatPromptTemplate.from_messages([
            (
                "system", 
                "Use the CONTEXT below to answer the user's question. If not in context, say you don't know."),
            ("system", "CONTEXT:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        
        
        doc_chain = create_stuff_documents_chain(llm=doc_llm, prompt=prompt_answer)
        
        return doc_chain
    
    def create_retrieval(self, history_retriever, docs_chain):
        retrieval_chain = create_retrieval_chain(
            retriever=history_retriever,
            combine_docs_chain=docs_chain
        )
        
        return retrieval_chain
    
    def get_chat_history(self, index_name: str):
        messages = RedisChatMessageHistory(
            session_id=index_name,
            redis_url=self.redis_url
        )
        
        return messages
    
    async def invoke(self, question: str, retrieval_chain, chat_history: RedisChatMessageHistory, system_prompt: str | list[str]):
        messages = chat_history.messages
        messages = [SystemMessage(content=system_prompt), *messages]
        
        result = await retrieval_chain.ainvoke({
            "input": question,
            "chat_history": messages
        })
        
        answer = result["answer"]
        
        await chat_history.aadd_messages([
            HumanMessage(content=question),
            AIMessage(content=answer)
        ])
        
        return answer
    
    async def stream_answer(self, question: str, retrieval_chain, chat_history: RedisChatMessageHistory, system_prompt: str | list[str]):
        messages = chat_history.messages
        messages = [SystemMessage(content=system_prompt), *messages]
       
        handler = StreamHandler()
        
        async for chunk in retrieval_chain.astream(
            {
                "input": question,
                "chat_history": messages
            },
            config={"callbacks": [handler]}
            ):
            pass
        
        print("".join(handler.title))
