import torch
import transformers

transformers.logging.set_verbosity_error()

def make_qa_s2s_batch(qa_list, tokenizer, max_len=64, max_a_len=256, device='cuda:0'):
    q_ls = [q for q, a in qa_list]
    a_ls = [a for q, a in qa_list]
    q_toks = tokenizer.batch_encode_plus(q_ls, max_length=max_len, padding='max_length', truncation=True)
    q_ids, q_mask = (
        torch.LongTensor(q_toks["input_ids"]).to(device),
        torch.LongTensor(q_toks["attention_mask"]).to(device),
    )
    a_toks = tokenizer.batch_encode_plus(a_ls, max_length=max_a_len, padding='max_length', truncation=True)
    a_ids, a_mask = (
        torch.LongTensor(a_toks["input_ids"]).to(device),
        torch.LongTensor(a_toks["attention_mask"]).to(device),
    )
    labels = a_ids[:, 1:].contiguous().clone()
    labels[a_mask[:, 1:].contiguous() == 0] = -100
    model_inputs = {
        "input_ids": q_ids,
        "attention_mask": q_mask,
        "decoder_input_ids": a_ids[:, :-1].contiguous(),
        "labels": labels}
    return model_inputs

def generate(
        question_doc,
        model,
        tokenizer,
        num_answers=1,
        num_beams=None,
        min_len=64,
        max_len=256,
        do_sample=False,
        temp=1.0,
        top_p=None,
        top_k=None,
        max_input_length=512,
        device='cuda:0'
    ):
    model_inputs = make_qa_s2s_batch([(question_doc, "A")], tokenizer, max_input_length, device=device)
    n_beams = num_answers if num_beams is None else max(num_beams, num_answers)
    generated_ids = model.generate(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        min_length=min_len,
        max_length=max_len,
        do_sample=do_sample,
        early_stopping=True,
        num_beams=1 if do_sample else n_beams,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        num_return_sequences=num_answers,
        decoder_start_token_id=tokenizer.bos_token_id,
    )
    return [tokenizer.decode(ans_ids, skip_special_tokens=True).strip() for ans_ids in generated_ids]

