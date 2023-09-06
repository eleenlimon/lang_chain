import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma



# Load PDF, DOCX, and TXT files as Langchain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


# Split data into chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# Create embeddings using OpenAIEmbeddings, and save to Chroma vector_store
def create_embeddings(index_name):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer


# Calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0004


# Clear chat history from streamlit
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == '__main__':
    import os

    # load OpenAI API key from .env
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.image('img.png')
    st.subheader('LLM Question-Answering Application')
    with st.sidebar:
        # text_input replacement for python-dotenv and .env -- for OpenAI API Key
        api_key = st.text_input("OpenAI API Key:", type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        # file upload widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # chunk size widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding file ... '):
                # writing the file from RAM to current directory
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                # create embeddings and return the Chroma vector_store
                vector_store = create_embeddings(chunks)

                # save vector_store in streamlit (persistent btwn reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully! :smile:')

    # user's question txt input widget
    q = st.text_input('Ask a question about the content of your file:')
    if q:        # user input question and hit enter
        if 'vs' in st.session_state:  # if  a file uploaded/embedded/split on vector_store)
            vector_store = st.session_state.vs
            # st.write(f'k:{k}')
            answer = ask_and_get_answer(vector_store, q, k)

            # text area widget for LLM Answer
            st.text_area('LLM Answer: ', value=answer)

            st.divider()

            # if no history, create it
            if 'history' not in st.session_state:
                st.session_state.history = ''

            # current question and answer
            value = f'Q: {q} \nA: {answer}'
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'

            # text area widget for chat history
            h = st.session_state.history
            st.text_area(label='Chat History', value=h, key='history', height=400)
