# https://gist.github.com/yuchenlin/a2f42d3c4378ed7b83de65c7a2222eb2#file-masked_word_prediction_bert-py
import numpy as np
import torch
from transformers import BertTokenizer, BertForMaskedLM
import re
import config as conf, consts as cn

device = conf.device

# TODO extract model specific functions to own module
#region MODEL
# pass model_name so function can be used e.g for a list of model names
def get_model(model_name = 'TurkuNLP/bert-base-finnish-cased-v1'):
    """Import model and tokenizer for masked LM task"""
    tokenizer = get_tokenizer(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model, tokenizer

def get_tokenizer(model_name = 'TurkuNLP/bert-base-finnish-cased-v1'):
    return BertTokenizer.from_pretrained(model_name)

def predict_masked_sent(model, tokenizer, tokenized_text):
    masked_index = tokenized_text.index(cn.mask) # NOTE takes first occurance of mask
    # convert textual tokens to numerical ids which correspond to token index in vocabulary
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device) # NOTE add batching?
    #tokens_tensor = tokens_tensor.to(device)

    with torch.no_grad():
        outputs = model(tokens_tensor) # TODO what is segments_tensors: https://medium.com/@dhartidhami/understanding-bert-word-embeddings-7dc4d2ea54ca
        predictions = outputs[0] # TODO batch?

    # logits to probabilities
    # select predictions for masked index
    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    return probs

def get_tokenized_sentence(sent, tokenizer):
    return tokenizer.tokenize("[CLS] %s [SEP]"%' '.join(sent))
#endregion

#region EVALUATE EXTENT OF BIAS
def get_association_score(model, tokenizer, sent, target_i, attribute_id):
    """TODO explain!"""
    # 1. Take a sentence with a target and attribute word
    masked_sent = split_sent(sent)
    # save target word id in bert vocab
    target_word_id = tokenizer.convert_tokens_to_ids(masked_sent[target_i])
    
    # 2. Mask the target word - target word is the ethnicity

    # handle out of vocab ethnicities - they are masked by splitting the word into tokens
    target_is_unk = target_word_id == tokenizer.convert_tokens_to_ids(cn.unk)
    if target_is_unk:
        tokenized_sent, target_word_id = mask_tokenized_word(target_i, masked_sent, tokenizer)
    else:
        masked_sent[target_i] = cn.mask
        tokenized_sent = get_tokenized_sentence(masked_sent, tokenizer)
    
    # 3. Obtain the probability of target word in the sentence
    # obtain the (prior) probability of the respective target word by using its vocabulary index
    target_prob = predict_masked_sent(model, tokenizer, tokenized_sent)[target_word_id]
    
    #For interpretation, a negative association between a target and an attribute means that the probability
    #of the target is lower than the prior probability, i.e. the probability of the target decreased through the
    #combination with the attribute. A positive association value means that the probability of the target in
    #creased through the combination with the attribute, with respect to the prior probability
    
    # 4. Mask both target and attribute word
    masked_sent[attribute_id] = cn.mask
    tokenized_sent = get_tokenized_sentence(masked_sent, tokenizer)
    
    # handle out of vocab ethnicities 
    if target_is_unk:
        tokenized_sent, target_word_id = mask_tokenized_word(target_i, masked_sent, tokenizer)

    # 5. Obtain the prior probability, i.e. the probability of the target word when the attribute is masked
    prior_prob = predict_masked_sent(model, tokenizer, tokenized_sent)[target_word_id]
    
    # 6. Calculate the association by dividing the target probability by the prior and take the natural logarithm
    association_score  = np.log(float(target_prob/prior_prob))
    return association_score

def mask_tokenized_word(word_i, sent, tokenizer):
    """
    Mask a tokenized word (TODO word pieces?) in a sentence.
    E.g. suomenruotsalaismies ['suomenruotsa', '##lais', '##mies']
    will be masked as ['[MASK],'##lais', '##mies']
    """
    tokenized_sent = get_tokenized_sentence(sent, tokenizer)
    # retrieve word root/stem; tokenize the word and select it's first segment
    word_root = tokenizer.tokenize(sent[word_i])[0]
    word_id = tokenizer.convert_tokens_to_ids(word_root) # word index in finbert vocab
    # mask the root of the word
    tokenized_sent[tokenized_sent.index(word_root)] = cn.mask
    return tokenized_sent, word_id

def split_sent(sent):
    """Split sentence into individual words and punctuation"""
    return re.findall(r"\w+|[^\w\s]", sent)

def process_scores(model, tokenizer, result):
    """TODO rename result"""

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