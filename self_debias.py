import math
import numpy as np
import torch
import score
import data
import context as c
import config
import consts
import json
import re

device = config.device
mask = consts.mask
unk = consts.unk


def get_scaling_vector(probs, decay_const=25):
    """
    Get vector for scaling probability distribution.
    Keep unbiased words' probability unchanged (1*probability)
    Scale biased words' probability down (e^decay_const*x) TODO decay_const??
    """
    def alpha(x): return torch.tensor(
        1) if x >= 0 else math.e**(decay_const * x)
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
            result[eth] = {'sents': [], 'terms': []}

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


def mask_targets(sentences, tokenizer):
    result = {}

    for eth in sentences:
        if eth != 'fin':
            result[eth] = {'sents': [], 'terms': []}

            for i, r in enumerate(sentences[eth]):
                sent = split_masked_sent(r[0]) if mask in r[0] else score.split_sent(r[0])
                ent_i = r[1]
                entity = sent[ent_i]
                entity_bert_i = tokenizer.convert_tokens_to_ids(entity)
                ent_is_unk = entity_bert_i == tokenizer.convert_tokens_to_ids(
                    unk)

                if ent_is_unk:
                    sent, _ = score.mask_tokenized_word(ent_i, sent, tokenizer)
                else:
                    sent[ent_i] = mask

                if sent not in result[eth]['sents']:
                    result[eth]['sents'].append(sent)
                if entity not in result[eth]['terms']:
                    result[eth]['terms'].append(entity)
    return result


def split_masked_sent(sent):
    """Custom sentence splitter to contain [MASK] as an individual token."""
    sent = sent.split()
    #sent[sent.index(mask)] = "@"

    # replace masked token with a single character
    # replace word pieces with single character
    word_pieces = []
    for i, piece in enumerate(sent):
        if piece == mask:
            sent[i] = "@"
        if piece.startswith("##"):
            word_pieces.append(piece)
            sent[i] = '#'

    # split the sentence into tokens
    res = re.findall(r"\w+|[^\w\s]", ' '.join(sent))

    #res[res.index('@')] = mask
    # add mask token back
    # add word pieces back
    word_pieces = iter(word_pieces)
    for i, token in enumerate(res):
        if token == "@":
            res[i] = mask
        if token == "#":
            res[i] = next(word_pieces)
    return res


def get_bert_and_new_probs(sentences, model, tokenizer, pref="", tokenize=True):
    # (sent, (word, old, new, difference))
    result = {}
    for eth in sentences:
        result[eth] = {}

        for i, sent in enumerate(sentences[eth]['sents']):
            inp = score.get_tokenized_sentence(sent, tokenizer) if tokenize else sent
            x_probs = score.predict_masked_sent(model, tokenizer, inp)

            # remove CLS and SEP if the sentence has already been tokenized
            if tokenize == False:
                sent = sent[1:len(sent)-1]

            sent2 = split_masked_sent(
                c.debiasing_template.format(sent=' '.join(sent)))
            inp2 = score.get_tokenized_sentence(sent2, tokenizer) if tokenize else ['[CLS]', *sent2, '[SEP]']
            sdb_probs = score.predict_masked_sent(model, tokenizer, inp2)

            new_probs = get_new_probs(x_probs, sdb_probs)

            # save probability distributions
            torch.save(x_probs, f"Results/raw/dists/{pref}{eth}_{i}_orig")
            torch.save(new_probs, f"Results/raw/dists/{pref}{eth}_{i}_new")

            # track changes of probability on term level
            temp = []
            for term in sentences[eth]['terms']:
                t_index = tokenizer.convert_tokens_to_ids(term)
                temp.append((term,                      # biased term
                            float(x_probs[t_index]),    # original probability
                            float(new_probs[t_index]),  # new probability
                            # difference between probabilities
                             float(x_probs[t_index] - new_probs[t_index])
                             ))
            result[eth][i] = (sent, *tuple(temp))
    return result


def get_bert_probs(sentences, model, tokenizer):
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


def save_antonym_probabilities(s_a, l_a, model, tokenizer):
    """Save raw probabilities of bias antonyms masked in sentence."""

    save_scores(get_bert_probs(s_a, model, tokenizer), 'ant_short_probs.json')
    save_scores(get_bert_probs(l_a, model, tokenizer), 'ant_long_probs.json')

    # cant use Results/raw/short or long because it saves only the association score, no probabilities


def main():
    short = data.get_context_sentences(c.context, c.context_t_i, c.context_a_i)
    long = data.get_context_sentences(
        c.context_long, c.context_long_t_i, c.context_long_a_i)
    short_ant = data.get_context_sentences(
        c.context, c.context_t_i, c.context_a_i, pos=True)
    long_ant = data.get_context_sentences(
        c.context_long, c.context_long_t_i, c.context_long_a_i, pos=True)

    s = mask_attributes(short)
    l = mask_attributes(long)
    s_a = mask_attributes(short_ant)
    l_a = mask_attributes(long_ant)
    
    model, tokenizer = score.get_model()  # finbert

    save_scores(get_bert_and_new_probs(
        s, model, tokenizer, pref="s_"), 'sdb_short.json')
    save_scores(get_bert_and_new_probs(
        l, model, tokenizer, pref="l_"), 'sdb_long.json')

    # save_antonym_probabilities(s_a, l_a, model, tokenizer)
    save_scores(get_bert_and_new_probs(
        s_a, model, tokenizer, pref="s_"), 'ant_short_probs.json')
    save_scores(get_bert_and_new_probs(
        l_a, model, tokenizer, pref="l_"), 'ant_long_probs.json')



if __name__ == "__main__":
    main()
