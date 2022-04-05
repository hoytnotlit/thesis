import pandas as pd
import numpy as np
# import matplotlib
import tikzplotlib
import matplotlib.pyplot as plt
#pd.set_option('display.max_rows', df.shape[0]+1)

unk = '[UNK]'

def get_df(scores, comp_scores, tokenizer):
    data_as_list = []

    for k in scores.keys():
        for i in range(len(scores[k])):
            biased = scores[k][i][0] > comp_scores[k][i][0]
            bias_is_unk = tokenizer.convert_tokens_to_ids(scores[k][i][2]) == tokenizer.convert_tokens_to_ids(unk)

            data_as_list.append((f'{k}',                # ethnicity
                                 scores[k][i][1],       # target word
                                 comp_scores[k][i][1],  # comparison target word (finnish)
                                 scores[k][i][3],       # entity
                                 scores[k][i][2],       # biased attribute
                                 round(scores[k][i][0], 4), # association score
                                 round(comp_scores[k][i][0], 4), # comparison association score
                                 biased,
                                 bias_is_unk
                                 ))

    df = pd.DataFrame(data=data_as_list, columns=['Ethnicity', 'Target', 'Comp. target', 'Entity',
                                                  'Bias', 'Association', 'Comp. association', 'Biased', 'Bias UNK'])
    return df

def get_nat_gen_means(df, file_name=None):
    # means of each ethnicity+gender
    grouped = df[['Association', 'Comp. association']].groupby(df['Ethnicity'])
    res = grouped.mean().round(4)
    if file_name != None:
        with open(f"Results/tables/{file_name}", "w") as file:
            file.write(res.to_latex())
    return res

def get_bias_means(df, no_unk = False, only_biased = False, file_name=None):
    # means for each bias
    # filter out biases that are not in BERT vocab
    if no_unk:
        df = df.loc[df['Bias UNK'] == False]
    # getting the biases which have a higher association in ethnic group
    if only_biased:
        df = df.loc[df['Association'] > df['Comp. association']]

    grouped = df.groupby(['Ethnicity', 'Bias'])
    res = grouped[['Association', 'Comp. association']].mean().round(4)#.sort_values(by=['Association'], ascending=False)

    if file_name != None:
        with open(f"Results/tables/{file_name}", "w") as file:
            file.write(res.to_latex())

    return res

def get_nat_means(df, file_name=None):
    # means of each ethnicity
    grouped = df[['Bias', 'Association', 'Comp. association']].groupby(df['Ethnicity'])
    res = grouped.mean().round(4)
    if file_name != None:
        with open(f"Results/tables/{file_name}", "w") as file:
            file.write(res.to_latex())
    return res

def get_nat_ent_means(df, file_name=None):
    # means of each entity in ethnicity
    res = df.groupby(['Ethnicity', 'Entity'])[['Association', 'Comp. association']].mean().round(4)
    if file_name != None:
        with open(f"Results/tables/{file_name}", "w") as file:
            file.write(res.to_latex())
    return res 

def get_eth_mean_chart(df, file_name="nat_mean.tex", save=True):
    data = [df['Association'].to_list(), df['Comp. association'].to_list()]
    ethnicities = df.T.columns.to_list()

    # if save:
    #     matplotlib.use("pgf")
    #     matplotlib.rcParams.update({
    #         "pgf.texsystem": "pdflatex",
    #         'font.family': 'serif',
    #         'text.usetex': True,
    #         'pgf.rcfonts': False,
    #     })

    X = np.arange(len(data[0]))
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(ethnicities, data[0], width = 0.25, label="Ethnicity")
    ax.legend()
    ax.bar(X + 0.25, data[1], width = 0.25, label="Finnish")
    ax.legend()

    plt.xlabel("Ethnic group")
    plt.ylabel("Association score mean")
    # plt.savefig(f'Results/visual/{file_name}')
    tikzplotlib.save(f'Results/charts/{file_name}')