name_dir = "Names/Top"
bias_dir = "Biases"

# tried different variations, current one is bad with "suomenruotsalais" 
# but the others work so I will keep like this
ethnicities = {
    'fin': 'suomalais',
    'swe': 'suomenruotsalais', # TODO rename fswe
    'roma': 'romani',
    'afro': 'somali', # TODO rename som
    'sami': 'saamelais',
    'rus': 'venäläis'
}

entities = ['nainen', 'mies', 'henkilö']


# top n (n=1) female and male names for ethnicity
names = {
    'fin': {'female': ['Päivi'], 'male': ['Mikko']},
    'swe': {'female': ['Laura'], 'male': ['Joel']},
    'sami': {'female': ['Elen'], 'male': ['Hugo']},
    'rus': {'female': ['Irina'], 'male': ['Nikolai']},
    'roma': {'female': ['Anneli'],'male': ['Veijo']},
    'afro': {'female': ['Isra'], 'male': ['Mohamed']}
 }

def get_context_sentences(context, target_i, attr_i, pos=False):
    result = { ethnicity: dict() for ethnicity in ethnicities }

    # handle opening correct bias files
    bias_file_base = "_biases.txt"
    if pos:
        bias_file_base = "_pos" + bias_file_base

    # iterate over each ethnicity
    for ethnicity in ethnicities:
        if ethnicity != 'fin':
            with open(f'{bias_dir}/{ethnicity}{bias_file_base}') as biases:
                result[ethnicity] = []
                result['fin'][ethnicity] = []
                
                for bias in biases:
                    # loop through both - names + entities
                    for ent in entities:
                        target = ethnicities[ethnicity] + ent
                        fin_target = ethnicities['fin'] + ent
                        
                        # capitalize first words
                        if target_i == 0:
                           target = target.capitalize()
                           fin_target = fin_target.capitalize()
                        if attr_i == 0:
                           bias = bias.capitalize()
                        
                        result[ethnicity].append((context.format(target=target, attribute=bias.strip()), 
                                                  target_i, 
                                                  attr_i, 
                                                  ent))
                        # add finnish groups for comparisons
                        result['fin'][ethnicity].append((context.format(target=fin_target, attribute=bias.strip()), 
                                                  target_i, 
                                                  attr_i, 
                                                  ent))

                    for gen in names[ethnicity]:
                        for name in names[ethnicity][gen]:
                            result[ethnicity].append((context.format(target=name, attribute=bias.strip()), 
                                                      target_i, 
                                                      attr_i, 
                                                      f'{gen} name'))
                    # finnish names for comparison
                    for gen in names['fin']:
                        for name in names['fin'][gen]:
                            result['fin'][ethnicity].append((context.format(target=name, attribute=bias.strip()), 
                                                      target_i, 
                                                      attr_i, 
                                                      f'{gen} name'))
    return result