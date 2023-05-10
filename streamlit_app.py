import streamlit as st
import datetime
import utils
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(page_title="Streamlit App", layout="wide")

# Define the function to render the Dashboard page
def render_overview():
    st.title("Avalanche Forecast Project Overview")
    # Add subheader with name and date
    st.subheader("Author: Jaymin West")
    
    # Add project description
    st.markdown("## Project Overview")
    st.markdown("The orginal idea for this project was to create a model that couuld predict the layers within a snow pack in a given region. Initially, I thought that I would be able to use weather data to achive this goal. However, it became evident that the data I was hoping to use was too sparse and what I thought was a datascience problem was actually a meteorology problem. As such, I had to adjust the scope of this project to focusing more on predicting the overall avalanche risk in a given area. ")
    st.markdown("### Step 1:")
    st.markdown("The area I chose to use was that covered by the Northwest Avalanche Center (NWAC). NWAC covers the western half of Washington and into Oregon. NWAC is a small part of Avalanche.org who host the websites for dozens of avalnche centers around the country. This is important as I built a webscraper to scrape all the archival data from NWAC's site and that webscraper could be expaned to work on all of the other avalanche centers. The code I wrote for the webscraper can be found in the webscraper.py file. It goes through a table of dates and gets the corresponding risk, area, and forecast for each date. The forecast comes in three sections, all of which go into varying levels of detail about the observed conditions and risks in the area")
    st.code("""
        import utils 
        import webscraper

        # Getting the avalanche data at Stevens Pass for the current season: 
        ws = webscraper.Webscraper("Stevens Pass")
        ws.open_archive_page()
        reports_data = ws.scrape_daily_reports()
        ws.to_csv(reports_data)

        data = utils.clean_raw_webscraper_data("output_data/Stevens_Pass_Current_Season_reports_data.csv")
        data = utils.add_weather_to_reports(data)
        data.to_csv("input_data/Stevens_Pass_Current_Season_reports_data.csv", index=False)
    """)
    st.markdown("### Step 2:")
    st.markdown("I was not expecting this to be one of the more time consuming section of the project but it was. It took me a long time trying to figure out how to best use my text data to predict the avalanche risk in the future. I figured I would have to do a lot of preprocessing to get my text into a state that would be useful for the computer. I took all of the text data, removed the stop words (including some custom stop words), lemmatized and lowercased everything, then combined it all into one column. All of these steps can be seen in the code below which is taken from my utils.py file.")
    
    st.code("""
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from gensim import corpora
    from gensim.utils import simple_preprocess
    import pandas as pd

    def preprocess_text_column(text):
        stop_words = set(stopwords.words('english'))
        # adding days of the week to stop words
        stop_words.update(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'])
        # adding months to stop words
        stop_words.update(['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                        'september', 'october', 'november', 'december'])
        lemmatizer = WordNetLemmatizer()
        
        try: 
            words = word_tokenize(text.lower())
            words = [w for w in words if not w in stop_words]
            words = [lemmatizer.lemmatize(w) for w in words]
            words = simple_preprocess(str(words), deacc=True)

            return ' '.join(words)
        except:
            return text
        
    def clean_raw_webscraper_data(fname):
        # Reading in the CSV file:
        all_data = pd.read_csv(fname)
        # Adding Column Names:
        all_data.columns = ['date', 'zone', 'overall_risk', 'above_treeline_risk', 'near_treeline_risk', 'below_treeline_risk', 'bottom_line_text', 'problem_type_text', 'forecast_discussion_text']
        # Adding a column for the combined text of all 3 text columns:
        all_data['combined_text'] = all_data['bottom_line_text'] + all_data['problem_type_text'] + all_data['forecast_discussion_text']
        # Converting date column to datetime:
        all_data['date'] = pd.to_datetime(all_data['date'])
        
        # Processing all text columns:
        text_coloumns = ['bottom_line_text', 'problem_type_text', 'forecast_discussion_text', 'combined_text']
        
        for column in text_coloumns:
            all_data[column] = all_data[column].apply(preprocess_text_column)

        all_data['zone'] = all_data['zone'].str.lower()
        all_data['overall_risk'] = all_data['overall_risk'].str.lower()

        # Mapping risk ratings to numbers:
        rating_mapping = {
            "extreme": 5.0,
            "high": 4.0,
            "considerable": 3.0,
            "moderate": 2.0,
            "low": 1.0,
            "no rating": 0.0
        }

        all_data['overall_risk'] = all_data['overall_risk'].map(rating_mapping)

        return all_data
    """)

    st.markdown("### Step 3:")
    st.markdown('My intiail plan for this project involved using an LSTM or some similar model to predict the avalanche risk for the next week. While I was able to do this, the results were lack luster and the project felt incomplete. I decided to switch gears and use GPT to make predictions based on the historical text data. This turned out to be a much more relevant and interesting outcome. ')
    st.code("""
            import keys
            from langchain import VectorDBQA, OpenAI
            from langchain.embeddings.openai import OpenAIEmbeddings
            from langchain.text_splitter import CharacterTextSplitter
            from langchain.vectorstores import Pinecone
            from langchain.document_loaders import DataFrameLoader
            import pandas as pd
            import pinecone

            api_key = keys.OPENAI_KEY

            # Loading the data:
            all_data = pd.read_csv('output_data/all_zones_all_data.csv')
            all_data = all_data.drop(columns=['bottom_line_text', 'problem_type_text', 'forecast_discussion_text'])
            all_data = all_data.dropna()

            # Loading the data:
            loader = DataFrameLoader(all_data, 'combined_text')
            documents = loader.load()

            # Splitting the data:
            text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
            docs = text_splitter.split_documents(documents)

            # Getting the embeddings:
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)

            # Using Pinecone as the vectorstore:
            pinecone.init(api_key=keys.PINECONE_KEY, environment='us-west1-gcp-free')
            index_name = "avalanche-reports"
            docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
            """)
    
    st.markdown('**Using LangChain To Load GPT with Custom Embeddings:**')
    st.code("""
    llm = OpenAI(openai_api_key=api_key)
    qa = VectorDBQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        vectorstore=docsearch,
        return_source_documents=False
    )
    query = "Rate the avalanche risk by day over the next week at Steven's Pass"
    results = qa.run(query)
    print(results)
    """)

    st.markdown("*Output: The avalanche risk is expected to increase over the next week at Steven's Pass, as warm and stormy weather will impact the area. Light rain and wet snow will fall on a primarily dry snowpack, and there is a greatest likelihood of encountering loose wet avalanches on south-facing slopes near treeline. Small wind slabs could still present a risk on steep exposed terrain, and there is a chance that an isolated terrain feature could sporadically release a glide avalanche. To minimize the risk, it is important to check the forecast, evaluate the conditions of the day, and make observations as you travel in the area. It is also important to avoid travel on steep terrain, choose terrain based on options that minimize exposure to loose wet avalanches, and find a thick crust topping the snow surface.*")
    st.markdown("**The output here can be see using the custom embedding data as it uses terms like 'near treeline' and 'south-facing slopes' which would not be present in the default GPT model.**")

    st.markdown("### Additional Steps:")
    st.markdown("**Creating an LSTM: ")
    st.code("""
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Embedding
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from gensim.models import Word2Vec
        from bokeh.plotting import figure, show, output_file

        # Load the dataframe
        # df = pd.read_csv('your_dataframe.csv')
        df = data

        # Select the features you want to use for training
        features = ['tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'pres', 'tsun']

        # Preprocess the data
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])  # Normalize the numerical features

        # Convert text tokens to numerical sequences
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df['tokens'])

        sequences = tokenizer.texts_to_sequences(df['tokens'])
        max_seq_length = max(len(seq) for seq in sequences)
        sequences = pad_sequences(sequences, maxlen=max_seq_length)

        tokenized_text = [str(tokens).split() for tokens in df['tokens']]
        # Train Word2Vec embeddings
        model = Word2Vec(sentences=tokenized_text, vector_size=50, window=7, min_count=7, workers=10)

        # Save the trained model
        model.save('models/first_word2vec.model')

        # Load pre-trained word embeddings (Word2Vec or GloVe)
        word_embeddings = Word2Vec.load('models/first_word2vec.model').wv

        # Create an embedding matrix
        embedding_dim = word_embeddings.vector_size
        vocab_size = len(tokenizer.word_index) + 1
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        for word, i in tokenizer.word_index.items():
            if word in word_embeddings.key_to_index:
                embedding_matrix[i] = word_embeddings[word]

        # Prepare the input and output data
        X = sequences
        y = df['overall_risk'].values

        # Define the LSTM model
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_seq_length, trainable=False))
        model.add(LSTM(16))
        model.add(Dense(1, activation='linear'))

        # Compile and train the model
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, y, epochs=20, batch_size=8)

    """)

    st.markdown("**Model Performance: **")
    HtmlFile = open("loss_plot.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height=700, width=700)



# Define the function to render the Forecast Discussion page
def render_forecast_discussion():
    st.title("Forecast Discussion")

    api_key = st.text_input("OpenAI API Key:")
    text_input = st.text_area("Enter Avalanche Query for %s:"%(datetime.date.today().strftime("%B %d, %Y")))
    if st.button("Submit"):
        # Pass the API key to the custom Langchain Model
        llm_output = utils.get_llm_prediction(api_key, text_input)
        st.write("Output:", llm_output)

    st.title("Find Most Similar Word: ")
    word_input = st.text_input("Enter Word:")
    if st.button("Submit Word"):
        similar_words = utils.find_most_similar_words(word_input)
        st.write("Most Similar Words By Embeddings:", similar_words)
        
# Create navigation menu
pages = ["Overview", "Forecast Discussion"]
selected_page = st.sidebar.selectbox("Select Page", pages)

# Render the selected page
if selected_page == "Overview":
    render_overview()
elif selected_page == "Forecast Discussion":
    render_forecast_discussion()
