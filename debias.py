import math
import torch
import score 

device = 'cuda'
mask = '[MASK]'
debiasing_template = 'Seuraava lause sisältää ennakkoluuloja: {sent}' # TODO more than 1 template?

# TODO idk if this is correct
def alpha(probs, decay_cons=25):
    new = torch.zeros_like(probs)
    for i, x in enumerate(probs):
        if x >= 0:
            new[i] = 1
        else:
            new[i] = math.e**(decay_cons * x)
    return new
    
def get_new_probs(x_probs, sdb_probs):
    # the difference between both distributions will be less than zero for such undesirable words
    # b/c sdb_probs will give higher probability for undesirable words!
    delta = x_probs - sdb_probs
    return alpha(delta) * x_probs

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
    return result