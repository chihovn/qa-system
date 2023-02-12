import os
from curses.ascii import US

import dotenv
import psycopg2
import torch
from matplotlib.style import use
from tqdm.notebook import tqdm
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

dotenv.load_dotenv()

DBNAME=os.getenv("DBNAME")
HOST=os.getenv("HOST")
PORT=os.getenv("PORT")
USER=os.getenv("DB_USER")
PWD=os.getenv("PASSWORD")
TB_CLIENT=os.getenv("TB_CLIENT")
TB_WIKI=os.getenv("TB_WIKI")
MSD_WIKI = bool(os.getenv("MSD_WIKI"))

def query_qa_dense_index(
        question: str, 
        qs_model, 
        qs_tokenizer, 
        device, 
        limit_doc =3, 
):
    """
    This function will receive a question, the question will be embeded
    and query to the database (only wiki database) to get relevant documents

    :param
        question: str
                user's question
        qs_model: PretrainedModel
                A Language Model is used to encode the `question`
        qs_tokenizer: PretrainedTokenizer
                A Tokenizer is used to create input for `qs_model`
        device: torch.device
                specify which gpu(s) will be used
    :return
        document_obj: List[Document]
                List of Document object (see `Document` class in `utils/schema.py`)
                [document_1: Document, document_2:Document, ..., document_n_results: Document]
    """
    question_embd = get_embds_qs(qs_model, qs_tokenizer, question, device= device)
    documents_wiki = query_embd(
                        question_embd, 
                        limit_doc=limit_doc, 
                    ) 
    document_obj_wiki = _create_list_doc(documents=documents_wiki, client=False)
    return document_obj_wiki

def query_qa_dense_index_client(
        question: str, 
        qs_model, 
        qs_tokenizer, 
        device, 
        limit_doc_wiki = 1,
        limit_doc_client = 3, 
        client = "msd"
):
    """
    This function will receive a question, the question will be encoded
    and queried to the database (wiki, client or both) to get relevant documents

    :param
        question: str
                user's question
        qs_model: PretrainedModel
                A Language Model is used to encode the `question`
        qs_tokenizer: PretrainedTokenizer
                A Tokenizer is used to create input for `qs_model`
        device: torch.device
                specify which gpu(s) will be used
        client: str
                ...
    :return
        document_obj: List[Document]
                List of Document object (see `Document` class in `utils/schema.py`)
                [document_1: Document, document_2: Document, ..., document_n_results: Document]
    """
    question_embd = get_embds_qs(qs_model,
                                 qs_tokenizer,
                                 question,
                                 device=device)

    document_obj = []

    if MSD_WIKI:
        if client == 'msd':
            documents_client = query_embd_db_client(question_embd, limit_doc_client)
            document_obj += _create_list_doc(documents=documents_client, client=True)
        documents_wiki = query_embd(question_embd, limit_doc_wiki)
        document_obj += _create_list_doc(documents=documents_wiki, client=False)

        return document_obj

    else:

        if client == 'msd':
            documents_client = query_embd_db_client(question_embd, limit_doc_client)
            document_obj += _create_list_doc(documents=documents_client, client=True)
        else:
            documents_wiki = query_embd(question_embd, limit_doc_wiki)
            document_obj += _create_list_doc(documents=documents_wiki, client=False)

        return document_obj

def _create_list_doc(
        documents, 
        client = False
    ): 
        """
        Convert A list of document tuples (output of querying from database) to
        a list of answer obejct (see Document class at `utils/schema.py`)

        :param
            documents: List[tuple]
                    List of tuples contains queried output from database
            client: boolean
                    process for client tuple or wiki tuple
                    because queried outputs from these 2 tables
                    are different from each other

        :return
            list of Document object
        """
        document_object = []
        if client: 
            for doc in documents: 
                doc_obj = Document(
                    content=doc[1], 
                    id= doc[0], 
                )
                document_object.append(doc_obj)
        else: 
            for doc in documents: 
                doc_obj = Document(
                    content=doc[3], 
                    meta={
                        "title": doc[1], 
                        "name": doc[2], 
                    }, 
                    id= doc[0], 
                )
                document_object.append(doc_obj)
        return document_object

def compare_q_vs_as(
        question, answers, 
        qs_model, qs_tokenizer, 
        ctx_model, ctx_tokenizer, 
        pdist, device
        ):
    """
    query = "something"
    devices, _ = initialize_device_settings(use_cuda=True, multi_gpu=False)
    pdist = torch.nn.PairwiseDistance(p=2) # define distance calculation 
    qs_model, qs_tokenizer = load_model_qs(pretrain_name="vblagoje/dpr-question_encoder-single-lfqa-wiki", device=devices[0])
    ctx_model, ctx_tokenizer = load_model_ctx(pretrain_name="vblagoje/dpr-ctx_encoder-single-lfqa-wiki", device=devices[0])
    answer = generator.predict(
        query = query, 
        documents=documents,
        num_return_sequences=1
    )
    final_ans = compare_q_vs_as(query, answer, qs_model, qs_tokenizer, ctx_model, ctx_tokenizer, pdist, device)
    print(final_ans[0].answer)

    Compare question embedding vs answer embeddings and sort the list of answers that are closest to
    question embedding

    :param
        question: str
                User's question
        answers - List[Answer]
                List of answers
        qs_model: PretrainedModel
                A langauge model is used to encode `question`
        qs_tokenizer: PretrainedTokenizer
                A Tokenizer that creates inputs for `qs_model`
        ctx_model: PretrainedModel
                A language model is used to encode each answer in `answers`
        ctx_tokenizer: PretrainedTokenizer
                A Tokenizer that creates inputs for `ctx_model`
        pdist: torch.nn.Module
                A module is used to calculate L2 distance between
                question embedding and answer embeddings
        device: torch.device
                Specify gpu(s) device

    :return
       List of answers sorted that are close to question in order 
    """    
    ans_lst = [ans.answer for ans in answers['answers']]
    ans_obj = [ans for ans in answers['answers']]
    question_embd = get_embds_qs(qs_model, qs_tokenizer, question, device= device)
    answers_embd = get_embds_qs(ctx_model, ctx_tokenizer, ans_lst, device= device)
    for i in range(len(ans_obj)): 
        ans_obj[i].embedding = answers_embd[i]
        ans_obj[i].score = float(pdist(question_embd, answers_embd[i]))
    return sorted(ans_obj, key = lambda x: x.score, reverse=False)

def query_embd_db_client(
        embd, 
        limit_doc=3, 
    ):
    """
    Query documents from database given a question embedding. These queried documents
    are sematically similar to question embedding (`embd`)

    :param
        embd: torch.tensor    : embedding of question
    :return
        a list of tuples containing features we required in `SELECT` as:
        [
            (id: int, content: str), (...), (...)
        ]
    """
    embd = str(list(embd.cpu().detach().numpy().reshape(-1)))
    try:
        connection = psycopg2.connect(dbname=DBNAME,
                                      host=HOST,
                                      port=PORT,
                                      user=USER,
                                      password=PWD)
        cursor = connection.cursor()
        aemb_sql = f"""
                        SET LOCAL ivfflat.probes = 3;
                        SELECT id, content FROM {TB_CLIENT}
                        ORDER BY embedd <#> %s LIMIT %s;
                    """

        cursor.execute(aemb_sql,(embd, limit_doc))
        connection.commit()
        rows = cursor.fetchall()

        if connection: 
            cursor.close()
            connection.close()
        
        return rows
        
    except (Exception, psycopg2.Error) as error: 
        print("Failed query record from database {}".format(error))

def query_embd(
        embd, 
        limit_doc=3, 
    ):
    """
    Query embedding from wiki database, question embedding will be compare (inner product)
    with document embeddings in database

    :param
        embd: torch.tensor    : question embedding
    :return
        a list of tuples containing features we required in `SELECT` as:
        [
            (id: int, title: str, name: str, content: str), (...), (...)
        ]
    """
    embd = str(list(embd.cpu().detach().numpy().reshape(-1)))
    try:
        connection = psycopg2.connect(dbname=DBNAME,
                                      host=HOST,
                                      port=PORT,
                                      user=USER,
                                      password=PWD)
        cursor = connection.cursor()
        aemb_sql = f"""
                        SET LOCAL ivfflat.probes = 3;
                        SELECT id, title, name, content FROM {TB_WIKI}
                        ORDER BY embedd <#> %s LIMIT %s;
                    """
        cursor.execute(aemb_sql,(embd, limit_doc))
        connection.commit()
        rows = cursor.fetchall()

        if connection: 
            cursor.close()
            connection.close()
        
        return rows
        
    except (Exception, psycopg2.Error) as error: 
        print("Failed query record from database {}".format(error))

def load_model_qs(
    pretrain_name="vblagoje/dpr-question_encoder-single-lfqa-wiki", 
    device = torch.device("cuda:0")
    ):
    """
    This function will load pretrained model and tokenizer from Huggingface 
    
    :param
        pretrain_name: str    
                name of the pretrained model 
        device: torch.device
                specified gpu device(s)
    
    :return 
        qs_model: PretrainedLM    
                model to encode the question 
        qs_tokenizer: PretrainedTokenizer 
                Tokenizer to create inputs and put them to `qs_model`
    """
    qs_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(pretrain_name)
    qs_model = DPRQuestionEncoder.from_pretrained(pretrain_name)
    qs_model.to(device)
    
    return qs_model, qs_tokenizer

def get_embds_qs(
    model, 
    tokenizer, 
    text, 
    device
    ):
    """
    This function gets the embedding of the text (question) and use this embedding to compute 
    similarity with documents in database

    :param
        model: PretrainedLM   
                model to encode the text 
        tokenizer: PretrainedTokenizer    
                Tokenizer to create inputs and put them to model
        text: str 
                the text or question needs to be encoded 
        device: torch.device  
                use gpu to create embedding 
    
    :return 
        model_output['pooler_output']: torch.tensor([1, 768]) 
                embedding of the text 
    """
    # Tokenize sentences
    model.eval()
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    return model_output['pooler_output']