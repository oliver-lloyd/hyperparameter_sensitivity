import pandas as pd
import os
from numpy.random import choice
from sklearn.model_selection import KFold
from string import ascii_uppercase

if __name__ == '__main__':

    datasets = [x.split('.')[0] for x in os.listdir('../raw_edgelists')]
    knockout_proportion = 0.05

    for dataset in datasets:

        remaining_edgelist = pd.read_csv(f'../raw_edgelists/{dataset}.tsv', sep='\t', header=None)
        remaining_proportion = 1 + knockout_proportion # + proportion to include full graph in below loop

        num_to_knockout = round(knockout_proportion * len(remaining_edgelist))
        knockouts = pd.DataFrame()

        while remaining_proportion > knockout_proportion:
            remaining_proportion = round(remaining_proportion - knockout_proportion, 2)

            # Perform knockouts
            if remaining_proportion < 1:
                knockout_IDs = choice(remaining_edgelist.index, num_to_knockout, replace=False)
                knocked_edges = remaining_edgelist.loc[knockout_IDs]
                knockouts = knockouts.append(knocked_edges)
                remaining_edgelist.drop(knockout_IDs, inplace=True)
                remaining_edgelist.reset_index(inplace=True, drop=True)

            # Save data
            new_edgelist_name = f'{dataset}_{remaining_proportion}'
            sub_dir = f'{dataset}/{new_edgelist_name}'
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)
            remaining_edgelist.to_csv(f'{sub_dir}/{new_edgelist_name}.tsv', index=False, header=None, sep='\t')
            knockouts.to_csv(f'{sub_dir}/knockouts_{new_edgelist_name}.tsv', index=False, header=None, sep='\t')

            # Create 5 folds for cross validation
            splits = list(KFold(5).split(remaining_edgelist.index))
            for i, split in enumerate(splits):
                fold_label = ascii_uppercase[i]
                
                # Randomly divide the 20% split into 10%/10% test/val
                test_inds = choice(split[1], int(len(split[1])/2), replace=False)
                val_inds = [ind for ind in split[1] if ind not in test_inds]
                
                # Locate edges
                train = remaining_edgelist.loc[split[0]]
                test = remaining_edgelist.loc[test_inds]
                val = remaining_edgelist.loc[val_inds]

                # Save
                fold_dir = sub_dir + f'/{new_edgelist_name}_{fold_label}'
                if not os.path.exists(fold_dir):
                    os.mkdir(fold_dir)
                train.to_csv(f'{fold_dir}/train.txt', sep='\t', index=False, header=None)
                test.to_csv(f'{fold_dir}/test.txt', sep='\t', index=False, header=None)
                val.to_csv(f'{fold_dir}/valid.txt', sep='\t', index=False, header=None)


            