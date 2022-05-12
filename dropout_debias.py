import data, score, self_debias as sdb
import json
import torch

model_path = 'test-mlm/'

context = 'Tutkimusten mukaan tyypillinen {target} on usein {attribute}.'
context_t_i = 3
context_a_i = 6

context_long = 'Helsingissä asuva {target} haluaa olla Suomen ensimmäinen \
presidentti, jolla on {attribute} vanhempi.'
context_long_t_i = 2
context_long_a_i = 11

def run_bias_analysis():
    # short sentences
    short = data.get_context_sentences(context, context_t_i, context_a_i)
    #pos_result = data.get_context_sentences(context, context_t_i, context_a_i, pos=True)

    # long sentences
    long = data.get_context_sentences(context_long, context_long_t_i, context_long_a_i)
    #long_pos_result = data.get_context_sentences(context_long, context_long_t_i, context_long_a_i, pos=True)

    s = sdb.mask_attributes(short)
    l = sdb.mask_attributes(long)

    model, tokenizer = score.get_model(model_path)
    old_model, old_tokenizer = score.get_model() # model to compare to

    sdb.save_scores(get_bert_probs(s, model, tokenizer, old_model, old_tokenizer, "s_"), 'dro_short.json')
    sdb.save_scores(get_bert_probs(l, model, tokenizer, old_model, old_tokenizer, "l_"), 'dro_long.json')

    #save_scores(score.process_scores(model, tokenizer, result), "short.json")
    #save_scores(score.process_scores(model, tokenizer, pos_result), "pos_short.json")
    #save_scores(score.process_scores(model, tokenizer, long_result), "long.json")
    #save_scores(score.process_scores(model, tokenizer, long_pos_result), "pos_long.json")

def get_bert_probs(sentences, model, tokenizer, orig_model, orig_tokenizer, pref=""):
    result = {}

    for eth in sentences:
        result[eth] = {}
        
        for i, sent in enumerate(sentences[eth]['sents']):
            inp = score.get_tokenized_sentence(sent, tokenizer)
            new_probs = score.predict_masked_sent(model, tokenizer, inp)

            orig_inp = score.get_tokenized_sentence(sent, orig_tokenizer)
            orig_probs = score.predict_masked_sent(orig_model, orig_tokenizer, orig_inp)

            # save probability distributions
            torch.save(new_probs, f"Results/raw/dists/{pref}{eth}_{i}_dro")

            # track changes of probability on term level
            temp = []
            for term in sentences[eth]['terms']:
                t_index = tokenizer.convert_tokens_to_ids(term)
                orig_t_index = orig_tokenizer.convert_tokens_to_ids(term)
                temp.append((term,                                # biased term
                            float(orig_probs[orig_t_index]),      # original probability
                            float(new_probs[t_index]),            # new probability
                            float(orig_probs[orig_t_index] - new_probs[t_index])    # difference between probabilities
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