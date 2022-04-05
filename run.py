import data, score
import json

context = 'Tutkimusten mukaan tyypillinen {target} on usein {attribute}.'
context_t_i = 3
context_a_i = 6

context_long = 'Helsingissä asuva {target} haluaa olla Suomen ensimmäinen \
presidentti, jolla on {attribute} vanhempi.'
context_long_t_i = 2
context_long_a_i = 11

def run_bias_analysis():
    # short sentences
    result = data.get_context_sentences(context, context_t_i, context_a_i)
    pos_result = data.get_context_sentences(context, context_t_i, context_a_i, pos=True)

    # long sentences
    long_result = data.get_context_sentences(context_long, context_long_t_i, context_long_a_i)
    long_pos_result = data.get_context_sentences(context_long, context_long_t_i, context_long_a_i, pos=True)

    model, tokenizer = score.get_model()
    # TODO try multilingual bert in comparison

    save_scores(score.process_scores(model, tokenizer, result), "short.json")
    save_scores(score.process_scores(model, tokenizer, pos_result), "pos_short.json")
    save_scores(score.process_scores(model, tokenizer, long_result), "long.json")
    save_scores(score.process_scores(model, tokenizer, long_pos_result), "pos_long.json")

def save_scores(jscores, file_name):
    with open(f"Results/raw/{file_name}", "w") as f:
        json.dump(jscores[0], f)
    with open(f"Results/raw/comp_{file_name}", "w") as f2:
        json.dump(jscores[1], f2)

if __name__ == "__main__":
    run_bias_analysis()