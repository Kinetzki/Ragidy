from rag import Ragidy
from dotenv import load_dotenv
load_dotenv()
import asyncio

async def main():
    ReliaAgent = Ragidy("redis://localhost:6379")
    await ReliaAgent.ingest_pdf("sample_file.pdf", "relia_ai")

    pdf_vs = ReliaAgent.get_pdf_vector_store("relia_ai")

    history_retriever = ReliaAgent.create_history_retriever(pdf_vs)

    doc_chain = ReliaAgent.create_doc_chain()

    retrieval_chain = ReliaAgent.create_retrieval(
        history_retriever=history_retriever, 
        docs_chain=doc_chain)

    chat_history = ReliaAgent.get_chat_history("user_1")

    system_persona = "You are Relia Ai, a helpful and professional assistant. Answer concisely, in plain language but add a joyful tone to you responses. If the answer to a question is not in CONTEXT, say you do not know."

    while True:
        question = input("Enter Question: ")
        
        # result = await ReliaAgent.invoke(
        #     question=question,
        #     retrieval_chain=retrieval_chain,
        #     chat_history=chat_history,
        #     system_prompt=system_persona)
         
        # print(result)
        async for chunk in ReliaAgent.stream_answer(
            question=question,
            retrieval_chain=retrieval_chain,
            chat_history=chat_history,
            system_prompt=system_persona
        ):
            print(chunk, end="", flush=True)
        print()

asyncio.run(main())