# https://gist.github.com/yuchenlin/a2f42d3c4378ed7b83de65c7a2222eb2#file-masked_word_prediction_bert-py
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
import re
import config as conf, consts as cn

device = conf.device
#mask = '[MASK]'
#unk = '[UNK]'

# pass model_name so function can be used e.g for a list of model names
def get_model(model_name = 'TurkuNLP/bert-base-finnish-cased-v1'):
    # import tokenizer and model for masked LM task
    tokenizer = get_tokenizer(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model, tokenizer

def get_tokenizer(model_name = 'TurkuNLP/bert-base-finnish-cased-v1'):
    return BertTokenizer.from_pretrained(model_name)

def predict_masked_sent(model, tokenizer, tokenized_text):
    masked_index = tokenized_text.index(cn.mask)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to(device)

    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    # logits to probabilities
    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    return probs

def get_association_score(model, tokenizer, sent, target_i, attribute_id):
    # 1. Take a sentence with a target and attribute word
    masked_sent = split_sent(sent)
    # save target word id in bert vocab
    target_word_id = tokenizer.convert_tokens_to_ids(masked_sent[target_i])
    
    # 2. Mask the target word - target word is the ethnicity

    # handle out of vocab ethnicities - they are masked by splitting the word into tokens
    target_is_unk = target_word_id == tokenizer.convert_tokens_to_ids(cn.unk)
    if target_is_unk:
        tokenized_sent, target_word_id = mask_tokenized_eth(target_i, masked_sent, tokenizer)
    else:
        masked_sent[target_i] = cn.mask
        tokenized_sent = get_tokenized(masked_sent, tokenizer)
    
    # 3. Obtain the probability of target word in the sentence
    # obtain the (prior) probability of the respective target word by using its vocabulary index
    target_prob = predict_masked_sent(model, tokenizer, tokenized_sent)[target_word_id]
    
    #For interpretation, a negative association between a target and an attribute means that the probability
    #of the target is lower than the prior probability, i.e. the probability of the target decreased through the
    #combination with the attribute. A positive association value means that the probability of the target in
    #creased through the combination with the attribute, with respect to the prior probability
    
    # 4. Mask both target and attribute word
    masked_sent[attribute_id] = cn.mask
    tokenized_sent = get_tokenized(masked_sent, tokenizer)
    
    # handle out of vocab ethnicities 
    if target_is_unk:
        tokenized_sent, target_word_id = mask_tokenized_eth(target_i, masked_sent, tokenizer)

    # 5. Obtain the prior probability, i.e. the probability of the target word when the attribute is masked
    prior_prob = predict_masked_sent(model, tokenizer, tokenized_sent)[target_word_id]
    
    # 6. Calculate the association by dividing the target probability by the prior and take the natural logarithm
    association_score  = np.log(float(target_prob/prior_prob))
    return association_score

def mask_tokenized_eth(target_i, masked_sent, tokenizer):
    tokenized_sent = get_tokenized(masked_sent, tokenizer)
    # tokenize ethnicity
    target_word = tokenizer.tokenize(masked_sent[target_i])[0]
    target_word_id = tokenizer.convert_tokens_to_ids(target_word)
    # mask the first part of tokenized ethnicity
    tokenized_sent[tokenized_sent.index(target_word)] = cn.mask
    return tokenized_sent, target_word_id

def get_tokenized(masked_sent, tokenizer):
    text = "[CLS] %s [SEP]"%' '.join(masked_sent)
    return tokenizer.tokenize(text)

def split_sent(sent):
    return re.findall(r"\w+|[^\w\s]", sent)

def process_scores(model, tokenizer, result):
    scores = {}
    comp_scores = {}

    for key in result.keys():
        if key != 'fin':
            scores[key] = [(get_association_score(model, tokenizer,
                            res[0], res[1], res[2]),    # pass sentence, target index and attribute index 
                            split_sent(res[0])[res[1]], # ethnicity
                            split_sent(res[0])[res[2]], # bias
                            res[3])                     # entity
                                for res in result[key]]
            # finnish association scores for comparison
            comp_scores[key] = [(get_association_score(model, tokenizer,
                                comp_res[0], comp_res[1], comp_res[2]), # pass sentence, target index and attribute index 
                                 split_sent(comp_res[0])[comp_res[1]],  # ethnicity
                                 split_sent(comp_res[0])[comp_res[2]],  # bias
                                 comp_res[3])                           # entity
                                     for comp_res in result['fin'][key]]
    return scores, comp_scores