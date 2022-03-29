import os

name_dir = "Names/Top"
bias_dir = "Biases"

ethnicities = {
    'fin': 'suomalainen ',#'suomalais', 
    'swe': 'suomenruotsalainen ',#'suomenruotsalais' ,
    'roma': 'romani ',
    'afro': 'somalialainen ',
    'sami': 'saamelainen ',#'saamelais',
    'rus': 'venäläinen ' #'venäläis'
}
entities = ['nainen', 'mies', 'henkilö']



def get_context_sentences(context, target_i, attr_i, pos=False):
    result = {'fin': dict()}

    # handle opening correct bias files
    bias_file_base = "_biases.txt"
    if pos:
        bias_file_base = "_pos" + bias_file_base

    # iterate over each ethnicity
    for ethnicity in ethnicities:
        if ethnicity not in result:
            result[ethnicity] = {}

        if ethnicity != 'fin':
            with open(f'{bias_dir}/{ethnicity}{bias_file_base}') as biases:
                result[ethnicity] = []
                result['fin'][ethnicity] = []
                
                for bias in biases:
                    for ent in entities:
                        target = ethnicities[ethnicity] + ent
                        fin_target = ethnicities['fin'] + ent
                        
                        # capitalize first words
                        #if target_i == 0:
                        #    target = target.capitalize()
                        #    fin_target = fin_target.capitalize()
                        #if attr_i == 0:
                        #    bias = bias.capitalize()
                        
                        result[ethnicity].append((context.format(target=target, attribute=bias.strip()), 
                                                  target_i, 
                                                  attr_i, 
                                                  ent))
                        # add finnish groups for comparisons
                        result['fin'][ethnicity].append((context.format(target=fin_target, attribute=bias.strip()), 
                                                  target_i, 
                                                  attr_i, 
                                                  ent))
    return result