import numpy as np
import os, pdb, sys
from pathlib import Path
import configparser


if __name__ == '__main__':

    # read config files
    config = configparser.ConfigParser()
    config.sections()
    config.read('config.ini')

    # 1. feature processing
    if config['mode'].getboolean('process_feature') is True:
        for dataset in ['iemocap', 'iemocap', 'crema-d']:
            if config['feature']['feature'] == 'emobase':
                cmd_str = 'taskset 100 python3 feature_extraction/opensmile_feature_extraction.py --dataset ' + dataset
            else:
                cmd_str = 'taskset 100 python3 feature_extraction/pretrained_audio_feature_extraction.py --dataset ' + dataset
            cmd_str += ' --feature_type ' + config['feature']['feature']
            cmd_str += ' --data_dir ' + config['dir'][dataset]
            cmd_str += ' --save_dir ' + config['dir']['save_dir']
            
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
            cmd_str += ' --save_dir ' + config['dir']['save_dir']
            
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
        cmd_str += ' --save_dir ' + config['dir']['save_dir']
        cmd_str += ' --leak_layer first --model_learning_rate 0.0001'
        
        print('Traing Attack model')
        print(cmd_str)
        os.system(cmd_str)
    
    # 5. Evaluating attack model
    if config['mode'].getboolean('attack_evaluate') is True:
        if config['mode'].getboolean('privacy_preserve_adversarial'):
            cmd_str = 'taskset 100 python3 evaluate_attack/federated_attribute_attack.py --dataset ' + config['dataset']['private_dataset']
            cmd_str += ' --adv_dataset ' + config['dataset']['adv_dataset']
            cmd_str += ' --feature_type ' + config['feature']['feature']
            cmd_str += ' --dropout ' + config['model']['dropout']
            cmd_str += ' --model_type ' + config['model']['fed_model']
            cmd_str += ' --learning_rate ' + config[config['model']['fed_model']]['lr']
            cmd_str += ' --local_epochs ' + config[config['model']['fed_model']]['local_epochs']
            cmd_str += ' --num_epochs ' + config[config['model']['fed_model']]['global_epochs']
            cmd_str += ' --save_dir ' + config['dir']['save_dir']
            cmd_str += ' --leak_layer first '
            cmd_str += ' --privacy_preserve_adversarial ' + str(config['mode'].getboolean('privacy_preserve_adversarial'))
            
            cmd_str += ' --eval_undefended ' + str(config['mode'].getboolean('eval_undefended'))
            cmd_str += ' --perturb_norm ' + config['privacy_preserve']['norm']
            cmd_str += ' --eps ' + config['privacy_preserve']['eps']
            cmd_str += ' --eps_step ' + config['privacy_preserve']['eps_step']
            cmd_str += ' --max_iter ' + config['privacy_preserve']['max_iter']
            cmd_str += ' --targeted ' + config['mode']['targeted']
        else: 
            cmd_str = 'taskset 100 python3 evaluate_attack/federated_attribute_attack.py --dataset ' + config['dataset']['private_dataset']
            cmd_str += ' --adv_dataset ' + config['dataset']['adv_dataset']
            cmd_str += ' --feature_type ' + config['feature']['feature']
            cmd_str += ' --dropout ' + config['model']['dropout']
            cmd_str += ' --model_type ' + config['model']['fed_model']
            cmd_str += ' --learning_rate ' + config[config['model']['fed_model']]['lr']
            cmd_str += ' --local_epochs ' + config[config['model']['fed_model']]['local_epochs']
            cmd_str += ' --num_epochs ' + config[config['model']['fed_model']]['global_epochs']
            cmd_str += ' --save_dir ' + config['dir']['save_dir']
            cmd_str += ' --leak_layer first'
            cmd_str += ' --privacy_preserve_adversarial ' + str(config['mode'].getboolean('privacy_preserve_adversarial'))
        print('Evaluating Attack model')
        print(cmd_str)
        os.system(cmd_str)
