import pandas as pd
#pd.set_option('display.max_rows', df.shape[0]+1)

def get_df(scores, comp_scores):
    data_as_list = []

    for k in scores.keys():
        for i in range(len(scores[k])):
            biased = scores[k][i][0] > comp_scores[k][i][0]
            data_as_list.append((f'{k}', 
                                 scores[k][i][1], 
                                 comp_scores[k][i][1],
                                 scores[k][i][3],
                                 scores[k][i][2],
                                 scores[k][i][0], 
                                 comp_scores[k][i][0], 
                                 biased))

    df = pd.DataFrame(data=data_as_list, columns=['Nationality', 'Name', 'Comp. name', 'Entity',
                                                  'Bias', 'Association', 'Comp. association', 'Biased'])
    return df

def get_nat_gen_means(df):
    # means of each nationality+gender
    grouped = df[['Association', 'Comp. association']].groupby(df['Nationality'])
    return grouped.mean()

def get_bias_means(df):
    # means of each bias
    grouped = df.groupby(['Bias', 'Nationality'])
    return grouped[['Association', 'Comp. association']].mean().sort_values(by=['Association'], ascending=False)
