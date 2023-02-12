from utils.lfqa_utils import *
from utils.timeit import *
import torch
from query_process import retrieve
import logging
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from answer_generate import generate

# config logger
transformers.logging.set_verbosity_error()
logging.basicConfig(filename='log/log.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

#config device
#use_cuda = torch.cuda.is_available()
device_name =  'cpu'#'cuda:0' if use_cuda else 'cpu'
device = torch.device(device_name)

# model variable
qs_model = qs_tokenizer = model = tokenizer  = None

#constant
CONTEXT_LENGTH = 10

@timeit
def load_model():
    logger.info("Start loading models and tokenizers")
    print("Start loading models and tokenizers")
    global qs_model, qs_tokenizer, model, tokenizer
    #load query system model, tokenizer
    qs_model, qs_tokenizer = load_model_qs(device=device)
    #load bart model
    model_path = 'model/bart-base_0.pth'
    model_name = 'facebook/bart-base'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device_name)
    param_dict = torch.load(model_path, map_location =device_name)  # has model weights, optimizer, and scheduler states
    model.load_state_dict(param_dict["model"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("End loading models and tokenizers")
    print("End loading models and tokenizers")


@timeit
def retrieve_contexts(question: str) -> list:
    logger.info('Start retrieving contexts...')
    retrieve_result = retrieve(question, qs_model, qs_tokenizer, device, limit_doc=CONTEXT_LENGTH)
    logger.info('End retrieving contexts...')
    logger.info(f'Contexts : {retrieve_result}')
    if retrieve_result is None:
        return []
    return retrieve_result

@timeit
def generate_answer_from_context(question: str, contexts: list) -> str:
    logger.info('Start generating answers...')
    support_doc = '<P>' + "<P>".join(contexts)
    question_doc = "question: {} context: {}".format(question, support_doc)
    answers = generate(question_doc, model, tokenizer, device=device_name)
    logger.info('End generating answers...')
    logger.info(f'Answers : {answers}')
    return answers[0]

@timeit
def get_answer(question: str) -> str:
    logger.info(f'Receive question: {question}')
    contexts = retrieve_contexts(question)
    answer = generate_answer_from_context(question, contexts)
    return answer

@timeit
def main():
    load_model()
    while True:
        question = input('INPUT YOUR QUESTION: ')
        answer = get_answer(question)
        print(f"HERE IS THE ANSWER:")
        print('--- Start answer ---')
        print(answer)
        print('---End answer---')

        is_continue = input('Continue (y/n): ')
        if is_continue.lower() == 'n':
            break


if __name__ == "__main__":
    main()