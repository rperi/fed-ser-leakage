import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy


from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import numpy as np
import torch.nn as nn
import sys, os, shutil, pickle, argparse, pdb
import torch.nn.functional as F

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model'))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'utils'))

from training_tools import EarlyStopping, seed_worker, result_summary
from attack_model_forDefense import attack_model

from tqdm import tqdm

# some general mapping for this script
gender_dict = {'F': 0, 'M': 1}
leak_layer_dict = {'full': ['w0', 'b0', 'w1', 'b1', 'w2', 'b2'],
                   'first': ['w0', 'b0'], 'second': ['w1', 'b1'], 'last': ['w2', 'b2']}
leak_layer_idx_dict = {'w0': 0, 'w1': 2, 'w2': 4, 'b0': 1, 'b1': 3, 'b2': 5}

class WeightDataGenerator():
    def __init__(self, dict_keys, data_dict = None):
        self.dict_keys = dict_keys
        self.data_dict = data_dict

    def __len__(self):
        return len(self.dict_keys)

    def __getitem__(self, idx):
        data_file_str = self.dict_keys[idx]
        gender = gender_dict[self.data_dict[data_file_str]['gender']]
        tmp_data = (self.data_dict[data_file_str][weight_name] - weight_norm_mean_dict[weight_name]) / (weight_norm_std_dict[weight_name] + 0.00001)
        weights = torch.from_numpy(np.ascontiguousarray(tmp_data))
        tmp_data = (self.data_dict[data_file_str][bias_name] - weight_norm_mean_dict[bias_name]) / (weight_norm_std_dict[bias_name] + 0.00001)
        bias = torch.from_numpy(np.ascontiguousarray(tmp_data))
        return weights, bias, gender

def evaluate(model, data_loader, loss_func):
    
    model.eval()
    step_outputs = []
    for batch_idx, data_batch in enumerate(tqdm(data_loader)):
        weights, bias, y = data_batch
        weights, bias, y = weights.to(device), bias.to(device), y.to(device)
        weights_bias = torch.cat((weights,bias.unsqueeze(dim=2)),2)
        logits = model(weights_bias)
        
        loss = loss_func(logits, y)

        predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
        pred_list = [predictions[pred_idx] for pred_idx in range(len(predictions))]
        truth_list = [y.detach().cpu().numpy()[pred_idx] for pred_idx in range(len(predictions))]
        step_outputs.append({'loss': loss.item(), 'pred': pred_list, 'truth': truth_list})

        torch.cuda.empty_cache()
    result_dict = result_summary(step_outputs, mode='test', epoch=0)  # epoch=0 just as a dummy value. Not used anywhere

    return result_dict

    
if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--adv_dataset', default='iemocap')
    parser.add_argument('--feature_type', default='apc')
    parser.add_argument('--learning_rate', default=0.001)
    parser.add_argument('--batch_size', default=10)
    parser.add_argument('--use_gpu', default=True)
    parser.add_argument('--num_epochs', default=200)
    parser.add_argument('--local_epochs', default=5)
    parser.add_argument('--device', default='0')
    parser.add_argument('--model_type', default='fed_sgd')
    parser.add_argument('--pred', default='emotion')
    parser.add_argument('--leak_layer', default='full')
    parser.add_argument('--dropout', default=0.2)
    parser.add_argument('--privacy_budget', default=None)
    parser.add_argument('--privacy_preserve_adversarial', default=False, action='store_true')
    parser.add_argument('--save_dir', default='/media/data/projects/speech-privacy')

    parser.add_argument('--eval_undefended', default=False, action='store_true')
    parser.add_argument('--perturb_norm', default='l_2')
    parser.add_argument('--eps', default=0.3)
    parser.add_argument('--eps_step', default=0.1)
    parser.add_argument('--max_iter', default=100)
    parser.add_argument('--prob_0', default=0.5)
    parser.add_argument('--targeted', default=False, action='store_true')
    parser.add_argument('--surrogate', default=False, action='store_true')
    parser.add_argument('--surrogate_dataset')  # TO be used only when above flag is set to True
    
    args = parser.parse_args()
    seed_worker(8)
    device = torch.device("cuda:"+str(args.device)) if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    model_setting_str = 'local_epoch_'+str(args.local_epochs) if args.model_type == 'fed_avg' else 'local_epoch_1'
    model_setting_str += '_dropout_' + str(args.dropout).replace('.', '')
    model_setting_str += '_lr_' + str(args.learning_rate)[2:]
    if args.privacy_budget is not None: model_setting_str += '_udp_' + str(args.privacy_budget)

    torch.cuda.empty_cache() 
    torch.multiprocessing.set_sharing_strategy('file_system')

    # 1. normalization tmp computations
    weight_norm_mean_dict, weight_norm_std_dict = {}, {}
    weight_sum, weight_sum_square = {}, {}
    for key in ['w0', 'w1', 'w2', 'b0', 'b1', 'b2']:
        weight_norm_mean_dict[key], weight_norm_std_dict[key] = 0, 0
        weight_sum[key], weight_sum_square[key] = 0, 0
    
    # the updates layer name and their idx in gradient file
    weight_name, bias_name = leak_layer_dict[args.leak_layer][0], leak_layer_dict[args.leak_layer][1]
    weight_idx, bias_idx = leak_layer_idx_dict[weight_name], leak_layer_idx_dict[bias_name]

    # 1.1 read all data and compute the tmp variables
    # Used for normalizing the inputs. Computes variables using shadow model parameters.
    shadow_training_sample_size, shadow_data_dict = 0, {}
    print('reading file %s' % str(Path(args.save_dir).joinpath('tmp_model_params', args.model_type, args.pred, args.feature_type, args.adv_dataset, model_setting_str)))
    for shadow_idx in range(5):
        print('reading shadow model %d' % (shadow_idx))
        for epoch in range(int(args.num_epochs)):
            adv_federated_model_result_path = Path(args.save_dir).joinpath('tmp_model_params', args.model_type, args.pred, args.feature_type, args.adv_dataset, model_setting_str, 'fold'+str(int(shadow_idx+1)))
            file_str = str(adv_federated_model_result_path.joinpath('gradient_hist_'+str(epoch)+'.pkl'))
            # if shadow_idx == 0 and epoch < 10:
            with open(file_str, 'rb') as f:
                adv_gradient_dict = pickle.load(f)
            for speaker_id in adv_gradient_dict:
                data_key = str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id
                gradients = adv_gradient_dict[speaker_id]['gradient']
                shadow_training_sample_size += 1
                
                # calculate running stats for computing std and mean
                shadow_data_dict[data_key] = {}
                shadow_data_dict[data_key]['gender'] = adv_gradient_dict[speaker_id]['gender']
                shadow_data_dict[data_key][weight_name] = gradients[weight_idx]
                shadow_data_dict[data_key][bias_name] = gradients[bias_idx]
                for layer_name in leak_layer_dict[args.leak_layer]:
                    weight_sum[layer_name] += gradients[leak_layer_idx_dict[layer_name]]
                    weight_sum_square[layer_name] += gradients[leak_layer_idx_dict[layer_name]]**2
            
    # 1.2 calculate std and mean
    for key in leak_layer_dict[args.leak_layer]:
        weight_norm_mean_dict[key] = weight_sum[key] / shadow_training_sample_size
        tmp_data = weight_sum_square[key] / shadow_training_sample_size - (weight_sum[key] / shadow_training_sample_size)**2
        weight_norm_std_dict[key] = np.sqrt(tmp_data)
        
    # 2. we evaluate the attacker performance on service provider training
    import pdb
    attack_model_result_path = Path(os.path.realpath(__file__)).parents[1].joinpath('results', 'attack', args.leak_layer, args.model_type, args.feature_type, model_setting_str    )
    loss = nn.NLLLoss().to(device)
    save_result_df = pd.DataFrame()
    eval_model = attack_model(args.leak_layer, args.feature_type)
    eval_model.load_state_dict(torch.load(str(attack_model_result_path.joinpath('private_'+args.dataset+'.pt'))))
    eval_model = eval_model.to(device)
    pdb.set_trace()

    if args.eval_undefended is True: # Set to true to evaluate benign performance
        pdb.set_trace()
        print("Evaluating benign attacker performance")
        # 2.1 we perform 5 fold evaluation, since we also train the private data 5 times
        for fold_idx in range(5):
            test_data_dict = {}
            for epoch in range(int(args.num_epochs)):
                row_df = pd.DataFrame(index=['fold'+str(int(fold_idx+1))])
                
                # Model related
                federated_model_result_path = Path(args.save_dir).joinpath('tmp_model_params', args.model_type, args.pred, args.feature_type, args.dataset, model_setting_str, 'fold'+str(int(fold_idx+1)))
                weight_file_str = str(federated_model_result_path.joinpath('gradient_hist_'+str(epoch)+'.pkl'))

                with open(weight_file_str, 'rb') as f:
                    test_gradient_dict = pickle.load(f)
                for speaker_id in test_gradient_dict:
                    data_key = str(fold_idx)+'_'+str(epoch)+'_'+speaker_id
                    gradients = test_gradient_dict[speaker_id]['gradient']
                    test_data_dict[data_key] = {}
                    test_data_dict[data_key]['gender'] = test_gradient_dict[speaker_id]['gender']
                    test_data_dict[data_key][weight_name] = gradients[weight_idx]
                    test_data_dict[data_key][bias_name] = gradients[bias_idx]

            dataset_test = WeightDataGenerator(list(test_data_dict.keys()), test_data_dict)
            test_loader = DataLoader(dataset_test, batch_size=20, num_workers=0, shuffle=False)
            test_result = evaluate(eval_model, test_loader, loss)
        
            row_df['acc'], row_df['uar'] = test_result['acc'], test_result['uar']
            save_result_df = pd.concat([save_result_df, row_df])
            save_result_df.to_csv(str(attack_model_result_path.joinpath('private_' + str(args.dataset) + '_result.csv')))
            del dataset_test, test_loader
            
        print("Performance on benign samples\n")
        print("Average Accuracy = {}, Average UAR = {}".format(np.mean(save_result_df['acc']), np.mean(save_result_df['uar'])))
    if args.privacy_preserve_adversarial:
        print("Evaluating adversarial perturbed attacker performance")
        model_setting_str +=  '_eps_' + str(args.eps)
        if args.surrogate:
            model_setting_str +=  '_surrogate_' + str(args.surrogate_dataset)
        if args.prob_0 != 0.5:
            model_setting_str +=  '_prob0_' + str(args.prob_0)
        if args.surrogate:
            attack_model_result_path = attack_model_result_path.joinpath(
                                    'adversarial_privacy_preserve_norm={}_eps={}_epsstep={}_targeted={}_usingSurrogate_{}_prob0={}'.format(args.perturb_norm,
                                    args.eps, 
                                    args.eps_step,
                                    args.targeted,
                                    args.surrogate_dataset,
                                    args.prob_0))
        else:
            attack_model_result_path = attack_model_result_path.joinpath(
                                    'adversarial_privacy_preserve_norm={}_eps={}_epsstep={}_targeted={}_prob0={}'.format(args.perturb_norm, 
                                    args.eps, 
                                    args.eps_step,
                                    args.targeted,
                                    args.prob_0))
        os.makedirs(attack_model_result_path, exist_ok=True)
        
        for fold_idx in range(5):
            print("Evaluating attacker model on fold {}".format(fold_idx))
            test_data_dict = {}
            for epoch in range(int(args.num_epochs)):
                row_df = pd.DataFrame(index=['fold'+str(int(fold_idx+1))])
                
                # Model related
                federated_model_result_path = Path(args.save_dir).joinpath('tmp_model_params_privacy', args.model_type, args.pred, args.feature_type, args.dataset, model_setting_str, 'fold'+str(int(fold_idx+1)))
                weight_file_str = str(federated_model_result_path.joinpath('gradient_hist_'+str(epoch)+'.pkl'))

                with open(weight_file_str, 'rb') as f:
                    test_gradient_dict = pickle.load(f)
                for speaker_id in test_gradient_dict:
                    data_key = str(fold_idx)+'_'+str(epoch)+'_'+speaker_id
                    gradients = test_gradient_dict[speaker_id]['gradient']
                    test_data_dict[data_key] = {}
                    test_data_dict[data_key]['gender'] = test_gradient_dict[speaker_id]['gender']
                    test_data_dict[data_key][weight_name] = gradients[weight_idx]
                    test_data_dict[data_key][bias_name] = gradients[bias_idx]

            dataset_test = WeightDataGenerator(list(test_data_dict.keys()), test_data_dict)
            test_loader = DataLoader(dataset_test, batch_size=80, num_workers=0, shuffle=False)
            test_result = evaluate(eval_model, test_loader, loss)
            row_df['acc'], row_df['uar'] = test_result['acc'], test_result['uar']
            save_result_df = pd.concat([save_result_df, row_df])
            del dataset_test, test_loader
            
        pdb.set_trace()
        row_df = pd.DataFrame(index=['average'])
        row_df['acc'], row_df['uar'] = np.mean(save_result_df['acc']), np.mean(save_result_df['uar'])
        save_result_df = pd.concat([save_result_df, row_df])
        save_result_df.to_csv(str(attack_model_result_path.joinpath('private_' + str(args.dataset) + '_result.csv')))
        print("Performance on privacy preserved samples\n")
        print("Average Accuracy = {}, Average UAR = {}".format(np.mean(save_result_df['acc']), np.mean(save_result_df['uar'])))
