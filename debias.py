import math
import numpy as np
import torch
import score 
import data, context as c, config
import json
import re

device = config.device
mask = '[MASK]'

def get_scaling_vector(probs, decay_cons=25):
    # apply scaling function
    # keep unbiased words the same (1*probability stays the same)
    # for biased words, scale probability down (e^decay_const*x)
    alpha = lambda x: torch.tensor(1) if x >=0 else math.e**(decay_cons * x)
    result = torch.tensor(list(map(alpha, probs))).to(device)
    return result
    
def get_new_probs(x_probs, sdb_probs):
    # the difference between both distributions will be less than zero for undesirable words
    # b/c sdb_probs will give higher probability for undesirable words!
    delta = x_probs - sdb_probs
    return get_scaling_vector(delta) * x_probs

def prep_data(sentences):
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

# TODO try masking target as well

# custom split to maintain [MASK] as individual token
def split_masked_sent(sent):
    # replace masked token with a single character
    sent = sent.split()
    mask_i = sent.index(mask)
    sent[mask_i] = "@"
    # split the sentence into tokens
    sent = ' '.join(sent)
    res = re.findall(r"\w+|[^\w\s]", sent)
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
            
            inp = score.get_tokenized(sent, tokenizer)
            x_probs = score.predict_masked_sent(model, tokenizer, inp)

            inp2 = score.get_tokenized(sent2, tokenizer)
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

def save_scores(data, file_name):
    with open(f"Results/raw/{file_name}", "w") as f:
        json.dump(data, f)

def main():
    short = data.get_context_sentences(c.context, c.context_t_i, c.context_a_i)
    long = data.get_context_sentences(c.context_long, c.context_long_t_i, c.context_long_a_i)

    s = prep_data(short)
    l = prep_data(long)

    model, tokenizer = score.get_model() # finbert

    save_scores(get_probabilities(s, model, tokenizer), 'sdb_short.json')
    save_scores(get_probabilities(l, model, tokenizer), 'sdb_long.json')

if __name__ == "__main__":
    main()