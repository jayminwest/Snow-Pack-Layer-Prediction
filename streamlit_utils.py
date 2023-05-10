import pandas as pd
import pinecone
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from langchain import VectorDBQA, OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import DataFrameLoader

def get_llm_prediction(api_key, query):
    """
    Uses the OpenAI api and the vector db on user's text
    """
    # api_key = keys.OPENAI_KEY # Uncomment
    # Loading the data:
    all_data = pd.read_csv('output_data/all_zones_all_data.csv')
    all_data = all_data.drop(columns=['bottom_line_text', 'problem_type_text', 'forecast_discussion_text'])
    all_data = all_data.dropna()

    # Loading the data:
    loader = DataFrameLoader(all_data, 'combined_text')
    documents = loader.load()
    # Initializing
    # Splitting the data:
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    docs = text_splitter.split_documents(documents)

    # Getting the embeddings:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Using Pinecone as the vectorstore:
    pinecone.init(api_key=keys.PINECONE_KEY, environment='us-west1-gcp-free')
    index_name = "avalanche-reports"
    docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

    llm = OpenAI(openai_api_key=api_key)
    qa = VectorDBQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        vectorstore=docsearch,
        return_source_documents=False
    )

    results = qa.run(query)
    return results

def find_most_similar_words(word):
    all_zones_df = pd.read_csv('input_data/Stevens_Pass_Current_Season_reports_data.csv')

    data = all_zones_df.drop(columns=['bottom_line_text', 'problem_type_text', 'forecast_discussion_text'])
    
    data['combined_text'] = data['combined_text'].astype(str)
    data['combined_text'] = data['combined_text'].apply(lambda x: word_tokenize(x.lower()))
    data.rename(columns={'combined_text': 'tokens'}, inplace=True)
    # Training the word to vec model:
    model = Word2Vec(sentences=data['tokens'], vector_size=100, window=5, min_count=1, workers=4)
    model.save('models/first_word2vec.model')

    model = Word2Vec.load('models/first_word2vec.model')
    # Create word embeddings lookup dictionary
    word_embeddings = {}

    for word in model.wv.index_to_key:
        word_embeddings[word] = model.wv[word]

    return model.wv.most_similar(word)