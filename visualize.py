from turtle import pos
import pandas as pd
import numpy as np
import tikzplotlib
import matplotlib.pyplot as plt
import consts as cn
import torch

des_l = 2  # decimal points to keep
tables_dir = "Results/tables/"
charts_dir = "Results/charts/"

# English translations for display
ethnicities_en = {
    'fin': 'Finnish',
    'fswe': 'Finnish-Swedish',
    'roma': 'Roma',
    'som': 'Somali',
    'sami': 'Sami',
    'rus': 'Russian'
}
entities_en = {'nainen': 'woman', 'mies': 'man', 'henkilö': 'person'}


def get_term_translations(file, lang='en'):
    bias_dir = f"Biases/{lang}"
    with open(f'{bias_dir}/{file}') as translations:
        res = [line.strip() for line in translations]
    return res


def save(path, result, index=True):
    with open(path, "w") as file:
        file.write(result.to_latex(index=index))

# region RAW DATAFRAMES


def get_df(scores, comp_scores, tokenizer, is_pos=False):
    data_as_list = []

    for k in scores.keys():
        eth_file = k if not is_pos else k + '_pos'
        translations = get_term_translations(f'{eth_file}_biases.txt')
        # TODO fix pos translations

        for i in range(len(scores[k])):
            biased = scores[k][i][0] > comp_scores[k][i][0]
            bias_is_unk = tokenizer.convert_tokens_to_ids(
                scores[k][i][2]) == tokenizer.convert_tokens_to_ids(cn.unk)
            association_score = round(scores[k][i][0], des_l)
            comp_association = round(comp_scores[k][i][0], des_l)
            ent = entities_en[scores[k][i][3]
                              ] if scores[k][i][3] in entities_en else scores[k][i][3]
            # each attribute has 5 sentences, use to get translation
            attr_en = translations[i//5]

            data_as_list.append((ethnicities_en[k],    # ethnicity
                                scores[k][i][1],       # target word
                                # comparison target word (finnish)
                                 comp_scores[k][i][1],
                                 ent,                   # entity
                                 scores[k][i][2],       # biased attribute
                                 attr_en,               # translation
                                 association_score,     # association score
                                 comp_association,      # comparison association score
                                 biased,                # association score > comparison score
                                 bias_is_unk            # term not in vocab
                                 ))
    # TODO add difference ?
    df = pd.DataFrame(data=data_as_list, columns=['Ethnicity', 'Target', 'Control target', 'Entity',
                                                  'Biased term', 'Translation', 'Association', 'Control association', 'Biased', 'Bias UNK'])
    return df


def get_sdb_df(debiased_data, t_i, tokenizer):
    data_as_list = []

    for k, v in debiased_data.items():
        translations = get_term_translations(f'{k}_biases.txt')

        for i in v:
            sent = v[i][0]
            for j, term in enumerate(v[i][1:]):
                bias_is_unk = tokenizer.convert_tokens_to_ids(
                    term[0]) == tokenizer.convert_tokens_to_ids(cn.unk)
                # get english translation for entity (I truly hate doing it like this ᕦ(ò_óˇ)ᕤ)
                ent_en = [
                    substring for substring in entities_en if substring in sent[t_i]]
                ent = entities_en[ent_en[0]] if len(ent_en) > 0 else sent[t_i]
                data_as_list.append(
                    (ethnicities_en[k], ent, translations[j], *term, bias_is_unk))
    df = pd.DataFrame(data=data_as_list, columns=[
                      'Ethnicity', 'Entity', 'Translation', 'Biased term', 'Original prob.', 'New prob', 'Difference', 'Bias UNK'])
    # add percentage change as column
    df = df.assign(Change=percentage_change(
        df['Original prob.'], df['New prob']).values).sort_values(by="Change", ascending=False)
    # rearrange columns
    cols = list(df.columns.values)
    cols.insert(cols.index("Biased term"), cols.pop(cols.index('Translation')))
    df = df[cols]
    return df


def get_ant_prob_df(data, t_i):
    data_as_list = []

    for k, v in data.items():
        translations = get_term_translations(f'{k}_pos_biases.txt')

        for i in v:
            sent = v[i][0]
            for j, term in enumerate(v[i][1:]):
                target = entities_en  # TODO
                ent = entities_en[sent[t_i]
                                  ] if sent[t_i] in entities_en else sent[t_i]
                data_as_list.append(
                    (ethnicities_en[k], ent, translations[j], *term))
    df = pd.DataFrame(data=data_as_list, columns=[
                      'Ethnicity', 'Entity', 'Antonym translation', 'Antonym', 'Antonym probability'])
    return df
# endregion

# region ASSOCIATION SCORES


def get_bias_means(df, no_unk=False, only_biased=False, file_name=None):
    """Retrieve DataFrame combining association score means grouped by biased terms"""

    # filter out biases that are not in BERT vocab
    if no_unk:
        df = df.loc[df['Bias UNK'] == False]
    # getting the biases which have a higher association in ethnic group
    if only_biased:
        df = df.loc[df['Association'] > df['Control association']]

    grouped = df.groupby(['Ethnicity', 'Biased term', 'Translation'])
    res = grouped[['Association', 'Control association']].mean().round(
        des_l).sort_values(by=['Ethnicity', 'Association'])  # , ascending=False)
    # TODO how can I sort so that group with largest association score is first??
    if file_name != None:
        with open(f"Results/tables/{file_name}", "w") as file:
            file.write(res.to_latex())
        # TODO dont do this graphs rn
        if False:
            df_dict = dict(tuple(res.groupby('Ethnicity')))
            for key in df_dict:
                save_bias_mean_chart(pd.DataFrame(
                    df_dict[key]), f'{key}_{file_name}')
    return res


def get_comb_bias_means(df, long_df, file_name, no_unk=False, only_biased=False):
    """Retrieve DataFrame combining association score means for short and long sentence"""

    if no_unk:
        df = df.loc[df['Bias UNK'] == False]

    df = df.copy()
    long_df = long_df.copy()

    # rename long columns
    long_df = long_df[['Association', 'Control association']].rename({"Association": "Long association", "Control association": "Long Control association"},
                                                                     axis="columns")
    df = df[['Ethnicity', 'Biased term', 'Translation',
             'Association', 'Control association']]
    result = pd.concat([df, long_df], axis=1, join="inner")
    result = result.groupby(['Ethnicity', 'Biased term', 'Translation'])
    result = result.mean().round(des_l).sort_values(
        by=['Ethnicity', 'Association'])

    if file_name != None:
        with open(f"Results/tables/{file_name}", "w") as file:
            file.write(result.to_latex())
    return result


def get_nat_means(df, file_name=None):
    """Retrieve DataFrame with association score means grouped by ethnic groups"""

    grouped = df[['Biased term', 'Association',
                  'Control association']].groupby(df['Ethnicity'])
    res = grouped.mean().round(des_l).reset_index()
    if file_name != None:
        with open(f"Results/tables/{file_name}", "w") as file:
            file.write(res.to_latex(index=False))
    return res


def get_ent_means(df, file_name=None):
    """Retrieve DataFrame with association score means grouped by entities and ethnic groups"""

    res = df.groupby(['Ethnicity', 'Entity'])[
        ['Association', 'Control association']].mean().round(des_l)
    if file_name != None:
        with open(f"Results/tables/{file_name}", "w") as file:
            file.write(res.to_latex())
    return res


def get_eth_mean_chart(df, file_name="nat_mean.tex", save=True, title=""):
    data = [df['Association'].to_list(), df['Control association'].to_list()]
    ethnicities = df['Ethnicity'].to_list()  # df.T.columns.to_list()

    X = np.arange(len(data[0]))
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(ethnicities, data[0], width=0.25, label="Ethnicity")
    ax.legend()
    ax.bar(X + 0.25, data[1], width=0.25, label="Finnish")
    ax.legend()

    plt.xlabel("Ethnic group")
    plt.ylabel("Association score mean")

    ax.set_xticklabels(ethnicities)
    plt.title(title)

    tikzplotlib.save(f'Results/charts/{file_name}')


def save_ent_mean_chart(df, file_name):
    df = df.groupby(['Entity'])[
        ['Association', 'Control association']].mean().round(des_l)
    data = [df['Association'].to_list(), df['Control association'].to_list()]
    entities = df.T.columns.to_list()

    X = np.arange(len(data[0]))
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(entities, data[0], width=0.25, label="Ethnicity")
    ax.legend()
    ax.bar(X + 0.25, data[1], width=0.25, label="Finnish")
    ax.legend()

    plt.xlabel("Entity")
    plt.ylabel("Association score mean")
    ax.set_xticklabels(entities)

    tikzplotlib.save(f'Results/charts/{file_name}')


def save_bias_mean_chart(df, file_name):
    df.index = df.index.droplevel()
    data = [df['Association'].to_list(), df['Control association'].to_list()]
    entities = df.T.columns.to_list()

    Y = np.arange(len(data[0]))
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.barh(entities, data[0], height=0.25, label="Ethnicity")
    ax.legend()
    ax.barh(Y + 0.25, data[1], height=0.25, label="Finnish")
    ax.legend()

    plt.xlabel("Association score mean")
    ax.set_xticklabels(entities)

    tikzplotlib.save(f'Results/charts/{file_name}')


def get_word_pair_comparison(df, pos_df, file_name):
    df = df.copy()
    pos_df = pos_df.copy()

    # filter out antonyms that have an out-of-vocab word pair
    biased_i = np.where(df['Bias UNK'] == False)[0]
    pos_df = pos_df.iloc[biased_i]

    # filter biased terms to contain only in-vocab words
    df = df.loc[df['Bias UNK'] == False]

    # TODO refactor
    # filter out out-of-vocab antonyms and their biased term pair
    for i in pos_df.index.to_list():
        if pos_df.loc[[i]]['Bias UNK'].values[0] == True:
            pos_df.drop(pos_df.loc[[i]].index, inplace=True)
            df.drop(df.loc[[i]].index, inplace=True)

    # rename pos columns
    # , 'Control association'"Control association":"Opposite Control association",
    pos_df = pos_df[['Biased term', 'Translation', 'Association']].rename({"Biased term": "Antonym", "Association": "Antonym association", "Translation": "Antonym translation"},
                                                                          axis="columns")
    # , 'Control association']]
    df = df[['Ethnicity', 'Biased term', 'Translation', 'Association']]
    result = pd.concat([df, pos_df], axis=1, join="inner")
    result = result.groupby(
        ['Ethnicity', 'Biased term', 'Translation', 'Antonym', 'Antonym translation'])
    result = result.mean().round(des_l).sort_values(
        by=['Ethnicity', 'Association'])

    if file_name != None:
        save(f"{tables_dir}{file_name}", result)
    return result
# endregion

# region DEBIAS SCORES


def get_sdb_means(df, file_name=None):
    # means of each ethnicity
    res = df.groupby(df['Ethnicity']).mean().sort_values(
        "Difference", ascending=False).reset_index()
    if 'Antonym probability' in res.columns:
        del res['Antonym probability']  # no need for this column
    res['Change'] = res['Change'].map('{0:.2f} %'.format)
    if file_name != None:
        with open(f"Results/tables/{file_name}", "w") as file:
            file.write(res.to_latex(index=False))
    return res


def get_top_n_changes(ant_comb, n=10, file_name=None, no_unk=False):
    if no_unk:
        ant_comb = ant_comb.loc[ant_comb['Bias UNK'] == False]

    res = ant_comb.head(n)
    res['Change'] = res['Change'].map('{0:.2f} %'.format)
    if file_name != None:
        save(f'{tables_dir}{file_name}', res, index=False)
    return res


def percentage_change(col1, col2):
    return ((col1 - col2) / col1) * 100


def get_sdb_ant_df(raw, ant_raw, file_name=None):
    res = pd.concat([raw, ant_raw[['Antonym', 'Antonym translation',
                    'Antonym probability']]], axis=1, join="inner")
    res.groupby(['Ethnicity', 'Biased term']).mean(
    ).sort_values(by=['Ethnicity', 'Difference'])
    #res['Change'] = res['Change'].map('{0:.2f} %'.format)

    if file_name != None:
        save(f"{tables_dir}{file_name}", res)
    return res


def get_sdb_ant_diff(comb_df, file_name=None):
    # assess whether new probs are less/more than antonym probs
    res = comb_df.groupby(['Ethnicity']).mean()[
        ['Original prob.', 'New prob', 'Antonym probability']]
    # add difference as column
    res['Original difference'] = res['Original prob.'] - \
        res['Antonym probability']
    res['New difference'] = res['New prob'] - res['Antonym probability']
    res.reset_index()
    if file_name != None:
        save(f"{tables_dir}{file_name}", res)
    return res
# endregion


def get_top_k_words(file, tokenizer, top_k=10):
    data_as_list = []
    probs = torch.load(f"Results/raw/dists/{file}")
    top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

    # get words
    for i, pred_idx in enumerate(top_k_indices):
        word = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        print("[MASK]: '%s'" % word, " | weights:", float(top_k_weights[i]))
        data_as_list.append((word, float(top_k_weights[i])))

    df = pd.DataFrame(data=data_as_list, columns=['Word', 'Probability'])
    return df
