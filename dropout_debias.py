import data
import score
import self_debias as sdb
import context as c
import json
import torch

model_path = 'test-mlm/'


def run_bias_analysis():
    # short sentences
    short = data.get_context_sentences(c.context, c.context_t_i, c.context_a_i)
    short_ant = data.get_context_sentences(
        c.context, c.context_t_i, c.context_a_i, pos=True)

    # long sentences
    long = data.get_context_sentences(
        c.context_long, c.context_long_t_i, c.context_long_a_i)
    long_ant = data.get_context_sentences(
        c.context_long, c.context_long_t_i, c.context_long_a_i, pos=True)

    s = sdb.mask_attributes(short)
    s_a = sdb.mask_attributes(short_ant)
    l = sdb.mask_attributes(long)
    l_a = sdb.mask_attributes(long_ant)

    model, tokenizer = score.get_model(model_path)
    old_model, old_tokenizer = score.get_model()  # model to compare to

    sdb.save_scores(get_bert_probs(s, model, tokenizer,
                    old_model, old_tokenizer, "s_"), 'dro_short.json')
    sdb.save_scores(get_bert_probs(s_a, model, tokenizer, old_model,
                    old_tokenizer, "s_", save_dist=False), 'dro_short_ant.json')
    sdb.save_scores(get_bert_probs(l, model, tokenizer,
                    old_model, old_tokenizer, "l_"), 'dro_long.json')
    sdb.save_scores(get_bert_probs(l_a, model, tokenizer, old_model,
                    old_tokenizer, "l_", save_dist=False), 'dro_long_ant.json')


def get_bert_probs(sentences, model, tokenizer, orig_model, orig_tokenizer, pref="", save_dist=True):
    result = {}

    for eth in sentences:
        result[eth] = {}

        for i, sent in enumerate(sentences[eth]['sents']):
            inp = score.get_tokenized_sentence(sent, tokenizer)
            new_probs = score.predict_masked_sent(model, tokenizer, inp)

            orig_inp = score.get_tokenized_sentence(sent, orig_tokenizer)
            orig_probs = score.predict_masked_sent(
                orig_model, orig_tokenizer, orig_inp)

            if save_dist:
                # save probability distributions
                torch.save(new_probs, f"Results/raw/dists/{pref}{eth}_{i}_dro")

            # track changes of probability on term level
            temp = []
            for term in sentences[eth]['terms']:
                t_index = tokenizer.convert_tokens_to_ids(term)
                orig_t_index = orig_tokenizer.convert_tokens_to_ids(term)
                temp.append((term,                                # biased term
                            # original probability
                             float(orig_probs[orig_t_index]),
                             # new probability
                             float(new_probs[t_index]),
                             # difference between probabilities
                             float(orig_probs[orig_t_index] - \
                                   new_probs[t_index])
                             ))
            result[eth][i] = (sent, *tuple(temp))
    return result


def save_scores(jscores, file_name):
    with open(f"Results/raw/deb_{file_name}", "w") as f:
        json.dump(jscores[0], f)
    with open(f"Results/raw/deb_comp_{file_name}", "w") as f2:
        json.dump(jscores[1], f2)


if __name__ == "__main__":
    run_bias_analysis()
