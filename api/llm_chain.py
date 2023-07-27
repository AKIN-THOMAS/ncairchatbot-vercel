import datetime
# from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory


import warnings
warnings.filterwarnings('ignore')

current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"


# load_dotenv()

persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)
llm = ChatOpenAI(model_name=llm_name, temperature=0)

template = """If the user says any greetings and says their name like "hello I'm bishop", 
always reply by greeting the person and saying his name  
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use as many sentences as needed to answer the question but Keep the answer as concise as possible. 
Always be nice at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=template,)

# Run chain
chain = RetrievalQA.from_chain_type(llm,
                                    retriever=vectordb.as_retriever(),
                                    return_source_documents=True,
                                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
# chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)


if __name__ == "__main__":
    print(chain.run("what is NCAIR?"))
