import json
import data, score, context as ctx

def run_bias_analysis():
    # short sentences
    #short_result = data.get_context_sentences(ctx.context, ctx.context_t_i, ctx.context_a_i)
    #short_pos_result = data.get_context_sentences(ctx.context, ctx.context_t_i, ctx.context_a_i, pos=True)

    # long sentences
    long_result = data.get_context_sentences(ctx.context_long, ctx.context_long_t_i, ctx.context_long_a_i)
    long_pos_result = data.get_context_sentences(ctx.context_long, ctx.context_long_t_i, ctx.context_long_a_i, pos=True)

    model, tokenizer = score.get_model()
    # TODO try multilingual bert in comparison
    # model, tokenizer = score.get_model("bert-base-multilingual-cased")

    #save_scores(score.process_scores(model, tokenizer, short_result), "short.json")
    #save_scores(score.process_scores(model, tokenizer, short_pos_result), "pos_short.json")
    save_scores(score.process_scores(model, tokenizer, long_result), "long.json")
    save_scores(score.process_scores(model, tokenizer, long_pos_result), "pos_long.json")

def save_scores(jscores, file_name):
    with open(f"Results/raw/{file_name}", "w") as f:
        json.dump(jscores[0], f)
    with open(f"Results/raw/comp_{file_name}", "w") as f2:
        json.dump(jscores[1], f2)

if __name__ == "__main__":
    run_bias_analysis()