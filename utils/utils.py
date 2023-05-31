import glob
from langchain.document_loaders import DirectoryLoader
import tiktoken
import pinecone
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
#from tqdm.auto import tqdm
from tqdm.autonotebook import tqdm #better for notebooks
from uuid import uuid4
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQAWithSourcesChain#includes the sources of the information used to answer the question
from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA #no sources get returned


def pdf_loader(path="files//",loader="unstructured"):
    """
    Unstructured reads the whole pdf as one Document, while pypdf creates a Document per page, which creates certain issues when pages dont contain a lot of text. 
    Unstructured in my opinion is better and can be better paired with recursive text splitter. pYPDF WOULD result to emptier chunks after recursive text splitter

    path:path/folder where the pdf files are located. We will load all of them. 
    loader: "unstructured or PyPDFloader

    documents: Our return data, it is a list of Document datatype. This datatype contains both text and metadata
    
    
    """

    if loader=="unstructured":
        loader = DirectoryLoader(path, glob="*.pdf", loader_cls=UnstructuredFileLoader)
        documents = loader.load()
    elif loader=="pypdf":
        loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()   
    else:
        raise ValueError("Unknown data loader") 
    
    print(f"No documents: {len(documents)}")
    return documents


def pdf_2_chunks(documents,tiktoken_model='cl100k_base',chunk_size=512,chunk_overlap=51,separators=['\n\n', '\n', ' ', '']):
    """
    Breaks down multiple documents(or just one) into chunks. By allowing an overlap between chunks, we try to alleviate the issue where importantint information gets cut in half

    returns the chunks of Documents that got sent
    """
    
    tokenizer = tiktoken.get_encoding(tiktoken_model)
    # create the function that calculates the token length
    def tiktoken_len(text):
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    text_splitter = RecursiveCharacterTextSplitter(
                                               chunk_size=chunk_size, 
                                               chunk_overlap=chunk_overlap,
                                               length_function=tiktoken_len,
                                               separators=separators
                                            )

    texts = text_splitter.split_documents(documents)
    return texts

def pinecone_init(api_key,environment,index_name,distance_metric="cosine",embed_dimension=1536):
    """
    This function initializes if needed(doesn't exist already) the index database in Pinecone.
    It then returns an index which will serve as our way to interact with Index/Vector Database
    """

    #initialize pinecone
    pinecone.init(
        api_key=api_key,  # find at app.pinecone.io
        environment=environment # next to api key in console
    )

    index_name = index_name #pinecone index(index is like a table for our embedding vectors)

    if index_name not in pinecone.list_indexes():#make sure the index doesnt exist
        # we create a new index
        pinecone.create_index(
            name=index_name,
            metric=distance_metric,
            dimension=embed_dimension # 1536 is the dimension of text-embedding-ada-002
        )

    #return index which will serve as our way to interact with Index/Vector Database
    index = pinecone.GRPCIndex(index_name) #grpc is more reliable than using simple index (we could have just used Index)

    print("Index status: ")
    print()
    print(index.describe_index_stats())

    return index

def load_chunks_2_pinecone(chunks,index,embed_api_key,embeddings_model_name='text-embedding-ada-002',batch_limit=100):
    """ 
    chunks:list of chunks or just one, that will be sent to our database
    embeddings: name of llm model we will use in order to generate embeddings
    index:pinecone index name, this is where our data will be sent
    """
    #we send the embeddings to the database, and add extra metadata
    embeddings = OpenAIEmbeddings(
        model=embeddings_model_name,
        openai_api_key=embed_api_key)


    batch_limit=batch_limit#this way we wont hit rate limits. We group our data before we send, otherwise if we made an api call for every chunk we would be capped from the rate limiter of the APIs we use

    todb_text = [] #text data we will embed
    todb_metadata = []#metadata for the corresponding text data


    for i, record in enumerate(tqdm(chunks)):


        todb_text.append(record.page_content)#text that will be embedded

        todb_metadata.append({
            "chunk":i,#we know from which chunk it originates
            "source": record.metadata["source"],#we know from pdf file it originates
            "text":record.page_content#it is necessary to add the actual text as metadata in order to use/display it when the closest neighbors are found
        })

        # if we have reached the batch_limit we can add texts to the database
        if len(todb_text) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(todb_text))]#create unique id for each embedding
            embeds = embeddings.embed_documents(todb_text)#embed text
            index.upsert(vectors=zip(ids, embeds, todb_metadata))#push our data to the index
            #reset the lists
            todb_text = []
            todb_metadata = []

    # if there are any left over we can add them to the database, since last if wont get triggered unless we surpass batch_limit
    if len(todb_text) > 0:
        ids = [str(uuid4()) for _ in range(len(todb_text))]
        embeds = embeddings.embed_documents(todb_text)
        index.upsert(vectors=zip(ids, embeds, todb_metadata))

    
def delete_pinecone_index(index_name):
    try:
        pinecone.delete_index(index_name)
    except:
        print("no info, make sure pinecone is initialized and database exists")

def pinecone_status(index):
    try:    
        print("Index status: ")
        print()
        print(index.describe_index_stats())
    except:
        print("no info, make sure pinecone is initialized and database exists")


def query_pinecone(index_name,embed_model_api_key,query,topk=3,embed_model_name='text-embedding-ada-002',qna_model="gpt-3.5-turbo",temperature=0.0,chain_type="stuff"):
    """
    This function performs two tasks. One, find the topk closest documents for the given query. Two, answer the query using those documents

    index_name = pinecone index name, we query this database
    embed_model_api_key = api key
    query = the question we will ask our model
    topk =  the number of closest documents (to our query) that we will return
    embed_model_name = open ai embedding model name
    qna_model = the open ai model we will use for qna
    temperature = 0, decrease model randomness/createvity, it limits model creativity which is important in making factual answers in qna. it doesnt guarantee that we will always have a factual answer but it helps
    chain type = TODO
    """
    
    embeddings = OpenAIEmbeddings(
    model=embed_model_name,
    openai_api_key=embed_model_api_key)
    
    # switch back to normal index for langchain
    index = pinecone.Index(index_name)
    vectorstore = Pinecone(
        index, embeddings.embed_query,"text" #text correlates to the text key of the metadata, thus it necessary when we send data to the index to have such key value
    )

    #find topk closest documents to our query
    vectorstore.similarity_search(
        query,  # our search query
        k=topk  # return top k most relevant docs
    )


    # completion llm
    llm = ChatOpenAI(
        openai_api_key=embed_model_api_key,
        model_name=qna_model,
        temperature=temperature#decrease model randomness/createvity, it limits model creativity which is important in making factual answers in qna. it doesnt guarantee that we will always have a factual answer but it helps
    )

    #llms can tell wrong things in a very convincing way, so we need to be careful with them aka hallucinations

    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=vectorstore.as_retriever()
    )
    #we give the users the citation, so he can check it himself where the info came from

    return qa_with_sources(query)
