import os
import pandas as pd

os.chdir('/home/jcsanguino10/local_citation_model/recommender-fusion-recsys/notebooks')
import multimodal_training as mt

import warnings
warnings.filterwarnings("ignore")


from itertools import chain, combinations

modalities = {
    'bpr': ('item_bpr_embedding', 'user_bpr_embedding'),
    'graph': ('item_graph_embedding', 'user_graph_embedding'),
    'text': ('item_text_embedding', 'user_text_embedding'),
}

sequence = 'user_sequence_embedding'  


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

final_combinations = []


for combo in powerset(modalities.keys()):
    item_embeds = [modalities[m][0] for m in combo]
    user_embeds = [modalities[m][1] for m in combo]
    final_combinations.append((item_embeds, user_embeds))


with_sequence = [ (items, users + [sequence]) for (items, users) in final_combinations ]
final_combinations.extend(with_sequence)


results_df = mt.test_multimodal_model(
    columns_combinations=final_combinations,
    encoding_dims=[720, 512],
    shared_dimensions=128,
    layers_per_modality=[256, 128],
    regularization_bpr_values=[0.005],
    verbose=False
)