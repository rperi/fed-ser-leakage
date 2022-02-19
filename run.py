import numpy as np
import os, pdb, sys
from pathlib import Path
import configparser
import sys

if __name__ == '__main__':

    config_file = sys.argv[1]
    save_dir = sys.argv[2]
    
    # read config files
    config = configparser.ConfigParser()
    config.sections()
    config.read(config_file)
    #config.read('config.ini')

    # 1. feature processing
    if config['mode'].getboolean('process_feature') is True:
        for dataset in ['iemocap', 'iemocap', 'crema-d']:
            if config['feature']['feature'] == 'emobase':
                cmd_str = 'taskset 100 python3 feature_extraction/opensmile_feature_extraction.py --dataset ' + dataset
            else:
                cmd_str = 'taskset 100 python3 feature_extraction/pretrained_audio_feature_extraction.py --dataset ' + dataset
            cmd_str += ' --feature_type ' + config['feature']['feature']
            cmd_str += ' --data_dir ' + config['dir'][dataset]
            cmd_str += ' --save_dir ' + save_dir
            
            print('Extract features')
            print(cmd_str)
            os.system(cmd_str)
    
    # 2. process training data
    if config['mode'].getboolean('process_training') is True:
        for dataset in ['msp-improv', 'iemocap', 'crema-d']:
            cmd_str = 'taskset 100 python3 preprocess_data/preprocess_federate_data.py --dataset ' + dataset
            cmd_str += ' --feature_type ' + config['feature']['feature']
            cmd_str += ' --data_dir ' + config['dir'][dataset]
            cmd_str += ' --save_dir ' + config['dir']['save_dir']
            cmd_str += ' --norm znorm'

            print('Process training data')
            print(cmd_str)
            os.system(cmd_str)

    # 3. Training SER model
    if config['mode'].getboolean('ser_training') is True:
        for dataset in [config['dataset']['private_dataset'], config['dataset']['adv_dataset']]:
            cmd_str = 'taskset 100 python3 train/federated_ser_classifier.py --dataset ' + dataset
            cmd_str += ' --feature_type ' + config['feature']['feature']
            cmd_str += ' --dropout ' + config['model']['dropout']
            cmd_str += ' --norm znorm --optimizer adam'
            cmd_str += ' --model_type ' + config['model']['fed_model']
            cmd_str += ' --learning_rate ' + config[config['model']['fed_model']]['lr']
            cmd_str += ' --local_epochs ' + config[config['model']['fed_model']]['local_epochs']
            cmd_str += ' --num_epochs ' + config[config['model']['fed_model']]['global_epochs']
            cmd_str += ' --save_dir ' + save_dir
            
            print('Traing SER model')
            print(cmd_str)
            os.system(cmd_str)

    # 4. Training attack model
    if config['mode'].getboolean('attack_training') is True:
        cmd_str = 'taskset 100 python3 train/federated_attribute_attack.py --dataset ' + config['dataset']['private_dataset']
        cmd_str += ' --norm znorm --optimizer adam'
        cmd_str += ' --adv_dataset ' + config['dataset']['adv_dataset']
        cmd_str += ' --feature_type ' + config['feature']['feature']
        cmd_str += ' --dropout ' + config['model']['dropout']
        cmd_str += ' --model_type ' + config['model']['fed_model']
        cmd_str += ' --learning_rate ' + config[config['model']['fed_model']]['lr']
        cmd_str += ' --local_epochs ' + config[config['model']['fed_model']]['local_epochs']
        cmd_str += ' --num_epochs ' + config[config['model']['fed_model']]['global_epochs']
        cmd_str += ' --save_dir ' + save_dir
        cmd_str += ' --leak_layer first --model_learning_rate 0.0001'
        if config['mode'].getboolean('normalize_disable'):
            cmd_str += ' --normalize_disable '
        print('Traing Attack model')
        print(cmd_str)
        os.system(cmd_str)
    
    # 5. Evaluating attack model
    if config['mode'].getboolean('attack_evaluate') is True:
        if config['mode'].getboolean('privacy_preserve_adversarial'):
            cmd_str = 'taskset 100 python3 evaluate_attack/federated_attribute_attack.py --dataset ' + config['dataset']['private_dataset']
            #cmd_str = 'taskset 100 python3 evaluate_attack/federated_attribute_attack_concept.py --dataset ' + config['dataset']['private_dataset']
            cmd_str += ' --adv_dataset ' + config['dataset']['adv_dataset']
            cmd_str += ' --feature_type ' + config['feature']['feature']
            cmd_str += ' --dropout ' + config['model']['dropout']
            cmd_str += ' --model_type ' + config['model']['fed_model']
            cmd_str += ' --learning_rate ' + config[config['model']['fed_model']]['lr']
            cmd_str += ' --local_epochs ' + config[config['model']['fed_model']]['local_epochs']
            cmd_str += ' --num_epochs ' + config[config['model']['fed_model']]['global_epochs']
            cmd_str += ' --save_dir ' + save_dir
            cmd_str += ' --leak_layer first '
            if config['mode'].getboolean('eval_undefended'):            
                cmd_str += ' --eval_undefended '
                #cmd_str += ' --eval_undefended ' + config['mode']['eval_undefended']
            cmd_str += ' --perturb_norm ' + config['privacy_preserve']['norm']
            cmd_str += ' --eps ' + config['privacy_preserve']['eps']
            cmd_str += ' --eps_step ' + config['privacy_preserve']['eps_step']
            cmd_str += ' --max_iter ' + config['privacy_preserve']['max_iter']
            if config['privacy_preserve']['prob_0']:
                cmd_str += ' --prob_0 ' + config['privacy_preserve']['prob_0']
            cmd_str += ' --privacy_preserve_adversarial '
            if config['mode'].getboolean('surrogate'):
                cmd_str += ' --surrogate '
                cmd_str += ' --surrogate_dataset ' + config['mode']['surrogate_dataset']
            if config['mode'].getboolean('targeted'):
                cmd_str += ' --targeted '
            if config['mode'].getboolean('normalize_disable'):
                cmd_str += ' --normalize_disable '
            if config['mode'].getboolean('privacy_preserve_random'):
                cmd_str += ' --privacy_preserve_random '

            #cmd_str += ' --privacy_preserve_adversarial ' + config['mode']['privacy_preserve_adversarial']
        elif config['mode'].getboolean('privacy_preserve_random'):
            cmd_str = 'taskset 100 python3 evaluate_attack/federated_attribute_attack.py --dataset ' + config['dataset']['private_dataset']
            #cmd_str = 'taskset 100 python3 evaluate_attack/federated_attribute_attack_concept.py --dataset ' + config['dataset']['private_dataset']
            cmd_str += ' --adv_dataset ' + config['dataset']['adv_dataset']
            cmd_str += ' --feature_type ' + config['feature']['feature']
            cmd_str += ' --dropout ' + config['model']['dropout']
            cmd_str += ' --model_type ' + config['model']['fed_model']
            cmd_str += ' --learning_rate ' + config[config['model']['fed_model']]['lr']
            cmd_str += ' --local_epochs ' + config[config['model']['fed_model']]['local_epochs']
            cmd_str += ' --num_epochs ' + config[config['model']['fed_model']]['global_epochs']
            cmd_str += ' --save_dir ' + save_dir
            cmd_str += ' --leak_layer first '
            if config['mode'].getboolean('eval_undefended'):         
                cmd_str += ' --eval_undefended '
                #cmd_str += ' --eval_undefended ' + config['mode']['eval_undefended']
            cmd_str += ' --privacy_preserve_random '
            if config['privacy_preserve']['noise_std']:
                cmd_str += ' --noise_std ' + config['privacy_preserve']['noise_std']
        else: 
            cmd_str = 'taskset 100 python3 evaluate_attack/federated_attribute_attack.py --dataset ' + config['dataset']['private_dataset']
            cmd_str += ' --adv_dataset ' + config['dataset']['adv_dataset']
            cmd_str += ' --feature_type ' + config['feature']['feature']
            cmd_str += ' --dropout ' + config['model']['dropout']
            cmd_str += ' --model_type ' + config['model']['fed_model']
            cmd_str += ' --learning_rate ' + config[config['model']['fed_model']]['lr']
            cmd_str += ' --local_epochs ' + config[config['model']['fed_model']]['local_epochs']
            cmd_str += ' --num_epochs ' + config[config['model']['fed_model']]['global_epochs']
            cmd_str += ' --save_dir ' + save_dir
            cmd_str += ' --leak_layer first'
            cmd_str += ' --eval_undefended '
        print('Evaluating Attack model')
        print(cmd_str)
        os.system(cmd_str)

    # 6. Training SER model with adversarially perturbed gradients or weights (privacy preserving)
    if config['mode'].getboolean('ser_training_privacy') is True:
        for dataset in [config['dataset']['private_dataset']]:
            cmd_str = 'taskset 100 python3 train/federated_ser_classifier_privacy.py --dataset ' + dataset
            cmd_str += ' --feature_type ' + config['feature']['feature']
            cmd_str += ' --dropout ' + config['model']['dropout']
            cmd_str += ' --norm znorm --optimizer adam'
            cmd_str += ' --model_type ' + config['model']['fed_model']
            cmd_str += ' --learning_rate ' + config[config['model']['fed_model']]['lr']
            cmd_str += ' --local_epochs ' + config[config['model']['fed_model']]['local_epochs']
            cmd_str += ' --num_epochs ' + config[config['model']['fed_model']]['global_epochs']
            cmd_str += ' --save_dir ' + save_dir
            cmd_str += ' --leak_layer first '
            cmd_str += ' --perturb_norm ' + config['privacy_preserve']['norm']
            cmd_str += ' --eps ' + config['privacy_preserve']['eps']
            cmd_str += ' --eps_step ' + config['privacy_preserve']['eps_step']
            cmd_str += ' --max_iter ' + config['privacy_preserve']['max_iter']
            if config['mode'].getboolean('targeted'):
                cmd_str += ' --targeted '
                if config['privacy_preserve']['prob_0']:
                    cmd_str += ' --prob_0 ' + config['privacy_preserve']['prob_0']
            if config['mode'].getboolean('surrogate'):
                cmd_str += ' --surrogate '
                cmd_str += ' --surrogate_dataset ' + config['mode']['surrogate_dataset']
                if config['mode'].getboolean('normalize'):
                    cmd_str += ' --adv_dataset ' + config['mode']['surrogate_dataset']
            if config['mode'].getboolean('normalize'):
                cmd_str += ' --normalize '
                cmd_str += ' --adv_dataset ' + config['mode']['adv_dataset']
            if config['mode'].getboolean('normalize_disable'):
                cmd_str += ' --normalize_disable '
            
            print('Traing SER model with privacy')
            print(cmd_str)
            os.system(cmd_str)
    
    # 7. Training SER model with randomly perturbed gradients or weights (baseline)
    if config['mode'].getboolean('ser_training_randomPerturb') is True:
        for data_idx, dataset in enumerate([config['dataset']['adv_dataset'], config['dataset']['private_dataset']]):
            cmd_str = 'taskset 100 python3 train/federated_ser_classifier_randomPerturb.py --dataset ' + dataset
            cmd_str += ' --feature_type ' + config['feature']['feature']
            cmd_str += ' --dropout ' + config['model']['dropout']
            cmd_str += ' --norm znorm --optimizer adam'
            cmd_str += ' --model_type ' + config['model']['fed_model']
            cmd_str += ' --learning_rate ' + config[config['model']['fed_model']]['lr']
            cmd_str += ' --local_epochs ' + config[config['model']['fed_model']]['local_epochs']
            cmd_str += ' --num_epochs ' + config[config['model']['fed_model']]['global_epochs']
            cmd_str += ' --save_dir ' + save_dir
            cmd_str += ' --leak_layer first '
            if config['privacy_preserve']['noise_std']:
                if data_idx == 1: # Private dataset. Use only one noise level to perturb
                    cmd_str += ' --noise_std ' + config['privacy_preserve']['noise_std']
                if data_idx == 0: # Shadow datasets. Use multiple noise levels to avoid overfitting of attacker
                    cmd_str += ' --noise_std ' + 'multiple'
                    
            print('Traing SER model with random perturbations')
            print(cmd_str)
            os.system(cmd_str)

    # 8. Training surrogate attack model
    if config['mode'].getboolean('attack_training_surrogate') is True:
        cmd_str = 'taskset 100 python3 train/federated_attribute_attack_surrogate.py --dataset ' + config['dataset']['private_dataset']
        cmd_str += ' --norm znorm --optimizer adam'
        cmd_str += ' --adv_dataset ' + config['dataset']['adv_dataset']
        cmd_str += ' --feature_type ' + config['feature']['feature']
        cmd_str += ' --dropout ' + config['model']['dropout']
        cmd_str += ' --model_type ' + config['model']['fed_model']
        cmd_str += ' --learning_rate ' + config[config['model']['fed_model']]['lr']
        cmd_str += ' --local_epochs ' + config[config['model']['fed_model']]['local_epochs']
        cmd_str += ' --num_epochs ' + config[config['model']['fed_model']]['global_epochs']
        cmd_str += ' --save_dir ' + save_dir
        cmd_str += ' --leak_layer first --model_learning_rate 0.0001'
        if config['mode'].getboolean('normalize_disable'):
            cmd_str += ' --normalize_disable '
        
        print('Traing Attack model')
        print(cmd_str)
        os.system(cmd_str)
