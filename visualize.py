# from turtle import pos
import pandas as pd
import numpy as np
import tikzplotlib
import matplotlib.pyplot as plt

unk = '[UNK]'
des_l = 2

# English translations for display
ethnicities_en = {
    'fin': 'Finnish',
    'fswe': 'Finnish-Swedish', 
    'roma': 'Roma',
    'som': 'Somali', 
    'sami': 'Sami',
    'rus': 'Russian'
}
entities_en = {'nainen':'woman', 'mies':'man', 'henkilö':'person'}

# TODO translate terms!
# in corresponsing en_*.txt files

def get_df(scores, comp_scores, tokenizer):
    data_as_list = []

    for k in scores.keys():
        for i in range(len(scores[k])):
            biased = scores[k][i][0] > comp_scores[k][i][0]
            bias_is_unk = tokenizer.convert_tokens_to_ids(scores[k][i][2]) == tokenizer.convert_tokens_to_ids(unk)
            association_score = round(scores[k][i][0], des_l)
            comp_association = round(comp_scores[k][i][0], des_l)
            ent = entities_en[scores[k][i][3]] if scores[k][i][3] in entities_en else scores[k][i][3]
            data_as_list.append((ethnicities_en[k],     # ethnicity
                                 scores[k][i][1],       # target word
                                 comp_scores[k][i][1],  # comparison target word (finnish)
                                 ent,                   # entity
                                 scores[k][i][2],       # biased attribute
                                 association_score,     # association score
                                 comp_association,      # comparison association score
                                 biased,                # association score > comparison score
                                 bias_is_unk            # term not in vocab
                                 ))
    # TODO add difference ?
    df = pd.DataFrame(data=data_as_list, columns=['Ethnicity', 'Target', 'Comp. target', 'Entity',
                                                  'Bias', 'Association', 'Comp. association', 'Biased', 'Bias UNK'])
    return df

def get_sdb_df(debiased_data):
    data_as_list = []

    for k, v in debiased_data.items():
        for i in v:
            sent = v[i][0]
            for term in v[i][1:]:
                data_as_list.append((k, sent, *term))

    df = pd.DataFrame(data=data_as_list, columns=['Ethnicity', 'Sentence', 'Biased term', 'Original prob.', 'New prob', 'Difference'])
    return df

def get_bias_means(df, no_unk = False, only_biased = False, file_name=None):
    # means for each bias
    # filter out biases that are not in BERT vocab
    if no_unk:
        df = df.loc[df['Bias UNK'] == False]
    # getting the biases which have a higher association in ethnic group
    if only_biased:
        df = df.loc[df['Association'] > df['Comp. association']]

    grouped = df.groupby(['Ethnicity', 'Bias'])
    res = grouped[['Association', 'Comp. association']].mean().round(des_l)

    if file_name != None:
        with open(f"Results/tables/{file_name}", "w") as file:
            file.write(res.to_latex())
        # TODO dont do this graphs rn
        if False:
            df_dict = dict(tuple(res.groupby('Ethnicity')))
            for key in df_dict:
                save_bias_mean_chart(pd.DataFrame(df_dict[key]), f'{key}_{file_name}')
    return res

def get_comb_bias_means(df, long_df, file_name, no_unk = False, only_biased = False):
    # combination bias means
    if no_unk:
        df = df.loc[df['Bias UNK'] == False]

    df = df.copy()
    long_df = long_df.copy()

    # rename long columns
    long_df = long_df[['Association', 'Comp. association']].rename({"Association": "Long association", "Comp. association":"Long comp. association"}, 
                axis="columns")
    df = df[['Ethnicity', 'Bias', 'Association', 'Comp. association']]
    result = pd.concat([df, long_df], axis=1, join="inner")
    result = result.groupby(['Ethnicity', 'Bias'])
    result = result.mean().round(des_l)

    if file_name != None:
        with open(f"Results/tables/{file_name}", "w") as file:
            file.write(result.to_latex())
    return result

def get_nat_means(df, file_name=None):
    # means of each ethnicity
    grouped = df[['Bias', 'Association', 'Comp. association']].groupby(df['Ethnicity'])
    res = grouped.mean().round(des_l)
    if file_name != None:
        with open(f"Results/tables/{file_name}", "w") as file:
            file.write(res.to_latex())
    return res

def get_ent_means(df, file_name=None):
    # means of each entity in ethnicity
    res = df.groupby(['Ethnicity', 'Entity'])[['Association', 'Comp. association']].mean().round(des_l)
    if file_name != None:
        with open(f"Results/tables/{file_name}", "w") as file:
            file.write(res.to_latex())
    return res 

def get_eth_mean_chart(df, file_name="nat_mean.tex", save=True, title=""):
    data = [df['Association'].to_list(), df['Comp. association'].to_list()]
    ethnicities = df.T.columns.to_list() 

    X = np.arange(len(data[0]))
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(ethnicities, data[0], width = 0.25, label="Ethnicity")
    ax.legend()
    ax.bar(X + 0.25, data[1], width = 0.25, label="Finnish")
    ax.legend()

    plt.xlabel("Ethnic group")
    plt.ylabel("Association score mean")

    ax.set_xticklabels(ethnicities)
    plt.title(title)
    
    tikzplotlib.save(f'Results/charts/{file_name}')

def save_ent_mean_chart(df, file_name):
    df = df.groupby(['Entity'])[['Association', 'Comp. association']].mean().round(des_l)
    data = [df['Association'].to_list(), df['Comp. association'].to_list()]
    entities = df.T.columns.to_list() 

    X = np.arange(len(data[0]))
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(entities, data[0], width = 0.25, label="Ethnicity")
    ax.legend()
    ax.bar(X + 0.25, data[1], width = 0.25, label="Finnish")
    ax.legend()

    plt.xlabel("Entity")
    plt.ylabel("Association score mean")
    ax.set_xticklabels(entities)
    
    tikzplotlib.save(f'Results/charts/{file_name}')

def save_bias_mean_chart(df, file_name):
    df.index = df.index.droplevel()
    data = [df['Association'].to_list(), df['Comp. association'].to_list()]
    entities = df.T.columns.to_list()

    Y = np.arange(len(data[0]))
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.barh(entities, data[0], height = 0.25, label="Ethnicity")
    ax.legend()
    ax.barh(Y + 0.25, data[1], height = 0.25, label="Finnish")
    ax.legend()

    plt.xlabel("Association score mean")
    ax.set_xticklabels(entities)

    tikzplotlib.save(f'Results/charts/{file_name}')

def get_word_pair_comparison(df, pos_df, file_name):
    df = df.copy()
    pos_df = pos_df.copy()

    # get the words that are biased
    biased_i = np.where(df['Bias UNK']==False)[0]

    # get opposite pairs for biased terms
    df = df.loc[df['Bias UNK'] == False]
    pos_df = pos_df.iloc[biased_i]

    # check if opposite pairs in vocab and remove
    for i in pos_df.index.to_list():
        if pos_df.loc[[i]]['Bias UNK'].values[0] == True:
            pos_df.drop(pos_df.loc[[i]].index, inplace=True)
            df.drop(df.loc[[i]].index, inplace=True)

    # rename pos columns
    pos_df = pos_df[['Bias', 'Association', 'Comp. association']].rename({"Bias": "Neut. bias", "Association": "Neut. association", "Comp. association":"Neut. comp. association"}, 
                axis="columns")
    df = df[['Ethnicity', 'Bias', 'Association', 'Comp. association']]
    result = pd.concat([df, pos_df], axis=1, join="inner")
    result = result.groupby(['Ethnicity', 'Bias', 'Neut. bias'])
    result = result.mean().round(des_l)

    if file_name != None:
        with open(f"Results/tables/{file_name}", "w") as file:
            file.write(result.to_latex())
    return result