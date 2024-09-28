import streamlit as st
import faiss
import transformers
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from torch import cuda, bfloat16
from langchain.embeddings import HuggingFaceEmbeddings #Embedding
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import law_split

docs = law_split.text_splitter('./data', type = '.docx')
docs = [ii for i in docs for ii in i]
documents =  []

for item in docs:
    title = item.split("\n\n")[0]
    page = Document(page_content=item,
    metadata = {"source": title})
    documents.append(page)

hf_token = st.secrets['hf_token']
st.set_page_config(page_title = "Vietnamese Legal Question Answering System", page_icon= "./app/static/Law.png", layout="centered", initial_sidebar_state="collapsed")

with open("./static/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(f"""
    <div class=logo_area>
        <img src="./app/static/Law.png"/>
    </div>
    """, unsafe_allow_html=True)
st.markdown(
    """
    <h1 style="text-align: center;">Vietnamese Legal Question Answering System</h1>
    """, 
    unsafe_allow_html=True
)

device = 0 if torch.cuda.is_available() else -1
if 'model_embedding' not in st.session_state:
    model_name = "bkai-foundation-models/vietnamese-bi-encoder"
    model_kwargs = {'device': 'cuda'} #cuda
    encode_kwargs = {'normalize_embeddings': False}
    st.session_state.model_embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
vectorstore = FAISS.from_documents(documents = documents, embedding = st.session_state.model_embedding)

# vectorstore = FAISS.load_local(
#     "faiss_db", st.session_state.model_embedding, allow_dangerous_deserialization=True
# )

if 'model' not in st.session_state:
    st.session_state.model = AutoModelForCausalLM.from_pretrained(
        'Viet-Mistral/Vistral-7B-Chat',
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=True,
        token = hf_token
    )
    
    st.session_state.tokenizer = AutoTokenizer.from_pretrained('Viet-Mistral/Vistral-7B-Chat', token = hf_token)
    generate_text = transformers.pipeline(
        model=st.session_state.model, tokenizer=st.session_state.tokenizer,
        return_full_text=False,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        # stopping_criteria=stopping_criteria,  # without this model rambles during chat
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # mex number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
        )
    st.session_state.llm = HuggingFacePipeline(pipeline=generate_text)

template = """Bạn là một chuyên gia trong lĩnh vực quản lý nhân sự và luật liên quan đến lao động và tiền lương ngoài lĩnh vực này vui lòng trả lời nằm ngoài lĩnh vực của bạn.
Sử dụng thông tin của các văn bản pháp luật sau để trả lời câu hỏi, nếu bạn không biết câu trả lời hoặc câu trả lời không có trong ngữ cảnh được cung cấp, chỉ cần nói rằng bạn không biết, không cố gắng tạo ra một câu trả lời.
{context}
Câu hỏi: {question}
Câu trả lời:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

def QA(query):
    qa_chain = RetrievalQA.from_chain_type(st.session_state.llm,
                                       retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                                       return_source_documents = True)    
    result = qa_chain({"query": query})
    return result['result'] + '\nDựa theo:\n\t' + '\n\t'.join(set([source.metadata['source'] for source in result['source_documents']]))


if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message['role'] == 'assistant':
        avatar_class = "assistant-avatar"
        message_class = "assistant-message"
        avatar = './app/static/AI.png'
    else:
        avatar_class = "user-avatar"
        message_class = "user-message"
        avatar = './app/static/human.jpg'
    st.markdown(f"""
    <div class="{message_class}">
        <img src="{avatar}" class="{avatar_class}" />
        <div class="stMarkdown">{message['content']}</div>
    </div>
    """, unsafe_allow_html=True)

if prompt := st.chat_input(placeholder='Xin chào, tôi có thể giúp được gì cho bạn?'):
    st.markdown(f"""
    <div class="user-message">
        <img src="./app/static/human.jpg" class="user-avatar" />
        <div class="stMarkdown">{prompt}</div>
    </div>
    """, unsafe_allow_html=True)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    respond = QA(prompt)

    st.markdown(f"""
    <div class="assistant-message">
        <img src="./app/static/AI.png" class="assistant-avatar" />
        <div class="stMarkdown">{respond}</div>
    </div>
    """, unsafe_allow_html=True)
    st.session_state.messages.append({'role': 'assistant', 'content': respond})


