import math
import numpy as np
import torch
import score 
import data, context as c, config, consts
import json
import re

device = config.device
mask = consts.mask#'[MASK]'
unk = consts.unk#'[UNK]'

def get_scaling_vector(probs, decay_const=25):
    """
    Get vector for scaling probability distribution.
    Keep unbiased words' probability unchanged (1*probability)
    Scale biased words' probability down (e^decay_const*x) TODO decay_const??
    """
    alpha = lambda x: torch.tensor(1) if x >=0 else math.e**(decay_const * x)
    result = torch.tensor(list(map(alpha, probs))).to(device)
    return result
    
def get_new_probs(x_probs, sdb_probs):
    # the difference between both distributions will be less than zero for undesirable words
    # b/c sdb_probs will give higher probability for undesirable words!
    delta = x_probs - sdb_probs
    return get_scaling_vector(delta) * x_probs

def mask_attributes(sentences):
    result = {}

    for eth in sentences:
        if eth != 'fin':
            result[eth] = {'sents':[], 'terms':[]}

            for i, r in enumerate(sentences[eth]):
                sent = score.split_sent(r[0])
                bias_term_i = r[2]
                bias_term = sent[bias_term_i]
                sent[bias_term_i] = mask

                if sent not in result[eth]['sents']:
                    result[eth]['sents'].append(sent)
                if bias_term not in result[eth]['terms']:
                    result[eth]['terms'].append(bias_term)
    return result

# TODO try masking target/ethnicity as well
def mask_targets(sentences, tokenizer):
    result = {}

    for eth in sentences:
        if eth != 'fin':
            result[eth] = {'sents':[], 'terms':[]}

            for i, r in enumerate(sentences[eth]):
                sent = score.split_sent(r[0])
                ent_i = r[1]
                entity = sent[ent_i]
                entity_bert_i = tokenizer.convert_tokens_to_ids(entity)
                ent_is_unk = entity_bert_i == tokenizer.convert_tokens_to_ids(unk)
                
                if ent_is_unk:
                    sent, _ = score.mask_tokenized_word(ent_i, sent, tokenizer)
                else:
                    sent[ent_i] = mask

                if sent not in result[eth]['sents']:
                    result[eth]['sents'].append(sent)
                if entity not in result[eth]['terms']:
                    result[eth]['terms'].append(entity)
    return result

# custom split to maintain [MASK] as individual token
def split_masked_sent(sent):
    # replace masked token with a single character
    sent = sent.split()
    sent[sent.index(mask)] = "@"
    # split the sentence into tokens
    res = re.findall(r"\w+|[^\w\s]", ' '.join(sent))
    # add mask token back
    res[res.index('@')] = mask
    return res

def get_probabilities(sentences, model, tokenizer):
    # (sent, (word, old, new, difference))
    result = {}

    for eth in sentences:
        result[eth] = {}
        
        for i, sent in enumerate(sentences[eth]['sents']):
            sent2 = split_masked_sent(c.debiasing_template.format(sent=' '.join(sent)))
            
            inp = score.get_tokenized_sentence(sent, tokenizer)
            x_probs = score.predict_masked_sent(model, tokenizer, inp)

            inp2 = score.get_tokenized_sentence(sent2, tokenizer)
            sdb_probs = score.predict_masked_sent(model, tokenizer, inp2)

            new_probs = get_new_probs(x_probs, sdb_probs)
            
            # track changes of probability on term level
            temp = []
            for term in sentences[eth]['terms']:
                t_index = tokenizer.convert_tokens_to_ids(term)
                temp.append((term,                      # biased term
                            float(x_probs[t_index]),    # original probability
                            float(new_probs[t_index]),  # new probability
                            float(x_probs[t_index] - new_probs[t_index])    # difference between probabilities
                            ))
            result[eth][i] = (sent, *tuple(temp))
    return result

# TODO rename
def get_only_probabilities(sentences, model, tokenizer):
    # (sent, (word, old, new, difference))
    result = {}

    for eth in sentences:
        result[eth] = {}
        
        for i, sent in enumerate(sentences[eth]['sents']):
            inp = score.get_tokenized_sentence(sent, tokenizer)
            probs = score.predict_masked_sent(model, tokenizer, inp)

            # track changes of probability on term level
            temp = []
            for term in sentences[eth]['terms']:
                t_index = tokenizer.convert_tokens_to_ids(term)
                temp.append((term,                      # biased term
                            float(probs[t_index]),      # original probability
                            ))
            result[eth][i] = (sent, *tuple(temp))
    return result

def save_scores(data, file_name):
    with open(f"Results/raw/{file_name}", "w") as f:
        json.dump(data, f)

# TODO get antonym probabilities
def load_antonym_probabilities(model, tokenizer):
    short_ant = data.get_context_sentences(c.context, c.context_t_i, c.context_a_i, pos=True)
    long_ant = data.get_context_sentences(c.context, c.context_t_i, c.context_a_i, pos=True)
    s_a = mask_attributes(short_ant)
    l_a = mask_attributes(long_ant)
    save_scores(get_only_probabilities(s_a, model, tokenizer), 'ant_short_probs.json')
    save_scores(get_only_probabilities(l_a, model, tokenizer), 'ant_long_probs.json')

    # make dict like {eth:[sent_tokenized, [masked_word, probability]]}
    # cant use Results/raw/short or long because it saves only the association score, no probabilities

def main():
    short = data.get_context_sentences(c.context, c.context_t_i, c.context_a_i)
    long = data.get_context_sentences(c.context_long, c.context_long_t_i, c.context_long_a_i)

    s = mask_attributes(short)
    l = mask_attributes(long)

    model, tokenizer = score.get_model() # finbert

    save_scores(get_probabilities(s, model, tokenizer), 'sdb_short.json')
    save_scores(get_probabilities(l, model, tokenizer), 'sdb_long.json')
    load_antonym_probabilities(model, tokenizer)

if __name__ == "__main__":
    main()