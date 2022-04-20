import math
import numpy as np
import torch
import score 

device = 'cuda:1'
mask = '[MASK]'
debiasing_template = 'Seuraava lause sisältää ennakkoluuloja: {sent}' # TODO more than 1 template?

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

def get_probabilities(sentences, model, tokenizer):
    result = []
    for s in sentences:
        sent = score.split_sent(s[0])
        t_index = tokenizer.convert_tokens_to_ids(sent[s[2]])
        sent[s[2]] = '[MASK]'
        sent2 = debiasing_template.format(sent=' '.join(sent)).split() # TODO fix split

        inp = score.get_tokenized(sent, tokenizer)
        x_probs = score.predict_masked_sent(model, tokenizer, inp)

        inp2 = score.get_tokenized(sent2, tokenizer)
        sdb_probs = score.predict_masked_sent(model, tokenizer, inp2)

        new_probs = get_new_probs(x_probs, sdb_probs)
        # TODO check how probability of attribute changes
        if new_probs[t_index] < x_probs[t_index]:
            print("difference", x_probs[t_index] - new_probs[t_index])
        # TODO mask also target??? association scores?
        # if im doing this I could just pass the masked sentence once and check for probabilities of all attribute words?
        result.append(new_probs)
    return result # TODO

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