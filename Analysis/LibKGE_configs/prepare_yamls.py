import yaml
from os import listdir
from numpy import arange

file_names = [name.split('.')[0] for name in listdir('templates') if name.endswith('.yaml')]

for template_name in file_names:
    
    # Load template YAML
    with open(f'templates/{template_name}.yaml') as f:
        config = yaml.safe_load(f)

    # Change meta settings
    config['search.num_workers'] = 16
    config['search.on_error'] = 'continue'
    config['train']['max_epochs'] = 500
    config['valid']['early_stopping']['patience'] = 20
    config['valid']['early_stopping']['min_threshold.metric_value'] = 0.25
    
    # Remove reciprocal relations as a model option
    method_name = template_name.split('_temp')[0]
    if method_name != 'conve': # ConvE depends on the reciprocal model. 
        config['ax_search']['parameters'][0]['value'] = method_name
        config['ax_search']['parameters'][0]['type'] = 'fixed'
        config['import'] = [method_name]
        del config['ax_search']['parameters'][0]['values']
        del config['reciprocal_relations_model.base_model.type']

    # Allow choice of all training types
    config['ax_search']['parameters'][2]['type'] = 'choice'
    config['ax_search']['parameters'][2]['values'] = ['1vsAll', 'KvsAll', 'negative_sampling']
    del config['ax_search']['parameters'][2]['value']

    # Add more PyTorch optimizers
    config['ax_search']['parameters'][3]['values'] += ['Adadelta', 'AdamW', 'SparseAdam', 'Adamax', 'ASGD', 'LBFGS', 'RMSprop', 'Rprop', 'SGD']

    # Add all loss funcs
    config['ax_search']['parameters'][4]['type'] = 'choice'
    config['ax_search']['parameters'][4]['values'] = ["bce", "bce_mean", "bce_self_adversarial", "margin_ranking", "ce", "kl", "soft_margin", "se"]
    del config['ax_search']['parameters'][4]['value']

    # Use smaller 'factor' arg for ReduceLROnPlateau (see PyTorch docs)
    # LibKGE devs choice of 0.95 seems way too high, is usually 0.5 to 0.1
    config['ax_search']['parameters'][8]['value'] = 0.5

    # Change bounds of ReduceLROnPlateau patience 
    config['ax_search']['parameters'][10]['bounds'] = [1, 10]

    # Set dropout lower bounds to 0 
    # TODO: what is negative dropout? maybe ask libkge devs
    config['ax_search']['parameters'][22]['bounds'] = [0.0, 0.5]
    config['ax_search']['parameters'][23]['bounds'] = [0.0, 0.5]

    # Remove model-specific hyperparameters
    if method_name in ['rescal', 'conve']:
        del config['ax_search']['parameters'][24]
    if method_name == 'conve':
        del config['ax_search']['parameters'][25]

    # Fix weight initialisation method subparameter values (uniform dist alpha, and normal dist std)
    config['ax_search']['parameters'][14]['type'] = 'fixed'
    config['ax_search']['parameters'][14]['value'] = 0.006737 # Logarithmic midpoint from LibKGE ICLR config bounds for this param
    del config['ax_search']['parameters'][14]['bounds']

    config['ax_search']['parameters'][15]['type'] = 'fixed'
    config['ax_search']['parameters'][15]['value'] = -0.5 # Middle value from LibKGE ICLR configs
    del config['ax_search']['parameters'][15]['bounds']

    # Write in dataset names and save 
    for dataset in ['UMLS-43', 'FB15k-237', 'WN18RR']:
        for proportion in arange(0.05, 1.05, 0.05):
            for fold in 'ABCDE':

                # Replace dataset.name
                dataset_name = f'{dataset}_{str(round(proportion, 2))}_{fold}'
                config['dataset.name'] = dataset_name

                # Save config
                config_name = f'{dataset_name}_{method_name}'
                with open(f'Configs/{dataset}/{config_name}.yaml', 'w') as f:
                    yaml.dump(config, f, default_flow_style=None)
