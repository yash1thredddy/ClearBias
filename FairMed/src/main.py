import copy
import json
import time
import torch
from einops import rearrange
import numpy as np
import pickle
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.globals import RESULTS_DIR, HF_NAMES, plot_line
from utils.dataset import TraceDataset
from evaluation import Evaluate, EvaluateBBQ, EvaluateBiasAsker, EvaluateTrace, EvaluateMMLU
from utils.logger import setup_logger

from utils.model import DNN, evaluate_dnn, train_dnn, plot_confusion_matrix

from utils.attack import pgd_attack_general

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument('--exp_suffix', type=str, default='')
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--trace_file', type=str, default='')
    parser.add_argument('--protect_attr', type=str, default='gender')
    
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--class_num', type=int, default=10)
    parser.add_argument('--num_mlps', type=int, default=9)

    parser.add_argument('--loss_type', type=str, default='KL', help='loss type')
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--bbq_cate', type=str)
    parser.add_argument('--eval_dataset_name', type=str)
    parser.add_argument('--noise_factor', type=float, default=5)
    parser.add_argument('--num_iter', type=int, default=10)
    parser.add_argument('--eps_iter', type=int, default=4) ### noise_factor/eps_iter
    parser.add_argument('--eval_task', type=str, help="just for eval")
    
    args = parser.parse_args()
    return args

class Ours:
    def __init__(self, args):
        self.args = args
        self.prepare(args)
        self.load_model(args)

    def prepare(self, args):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        exp_dir = os.path.join(RESULTS_DIR, "AdvDebias", args.model_name, args.eval_dataset_name, 
                               args.protect_attr, args.exp_suffix)
        os.makedirs(exp_dir, exist_ok=True)
        log_dir = os.path.join(exp_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        logger = setup_logger(work_dir=log_dir, logfile_name='log.txt')
        
        self.logger = logger
        self.exp_dir = exp_dir
        self.log_dir = log_dir

        self.logger.info(f"args: {args}")

        self.trace_dir = os.path.join(exp_dir, "trace")
        os.makedirs(self.trace_dir, exist_ok=True)
        self.probe_dir = os.path.join(exp_dir, "probe")
        os.makedirs(self.probe_dir, exist_ok=True)
        self.debias_dir = os.path.join(exp_dir, "debias")
        os.makedirs(self.debias_dir, exist_ok=True)
    def load_model(self, args):
        model_name_or_path = HF_NAMES[args.model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, add_bos_token=False)
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        tokenizer.pad_token = "</s>"
        tokenizer.unk_token = "<unk>"
        tokenizer.padding_side = "right"
        
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage = True, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        # define number of layers and heads
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.num_heads = num_heads
    
    def load_dataset(self):
        act_dataset = TraceDataset(self.trace_dir, args.trace_file, self.tokenizer, self.num_layers, self.num_heads,
                                   class_num=args.class_num, val_ratio=args.val_ratio, protect_attr=args.protect_attr)
        all_X_train, all_X_val, y_train, y_val = act_dataset.get_train_val_set()
        
        self.all_X_train = all_X_train
        self.all_X_val = all_X_val
        self.y_train = y_train
        self.y_val = y_val

    def run_trace(self):
        trace_start = time.time()
        self.logger.info(f"begin to run trace")
        
        evaluator = EvaluateTrace(self.model, self.tokenizer,
                self.args.trace_file, "trace", self.args.protect_attr)

        evaluator.evaluate()
        evaluator.save_results(self.trace_dir)
        self.logger.info(f"Trace consume time: {time.time()-trace_start}")
        self.trace_time = time.time()-trace_start
    def run_probe(self):
        args = self.args
        probe_start = time.time()
        self.logger.info(f"begin to run probe")
        self.logger.info(f"model device: {self.model.device}")

        probes, all_mlp_accs_np, all_mlp_f1s_np, all_mlp_kls_np, all_mlp_mses_np, all_mlp_cms_np =\
            self.train_dnn(self.all_X_train, self.all_X_val, self.y_train, self.y_val, self.model.device.index, class_num=args.class_num)
        self.logger.info(f"all_mlp_accs_np: {sorted(all_mlp_accs_np, reverse=True)}")
        self.logger.info(f"all_mlp_f1s_np: {sorted(all_mlp_f1s_np, reverse=True)}")
        self.logger.info(f"all_mlp_kls_np: {sorted(all_mlp_kls_np, reverse=False)}")
        self.logger.info(f"all_mlp_mses_np: {sorted(all_mlp_mses_np, reverse=False)}")
        top_mlps = np.argsort(all_mlp_f1s_np)[::-1][:args.num_mlps]

        plot_line(all_mlp_accs_np, name=f"{self.log_dir}/mlp_accs.png")
        plot_line(all_mlp_f1s_np, name=f"{self.log_dir}/mlp_f1s.png")
        plot_line(all_mlp_kls_np, name=f"{self.log_dir}/mlp_kls.png")
        plot_line(all_mlp_mses_np, name=f"{self.log_dir}/mlp_mses.png")

        for i, mlp in enumerate(all_mlp_cms_np):
            plot_confusion_matrix(mlp, name=f"{self.log_dir}/mlp_layer_{i}_cm.png")
        self.logger.info(f"Mlps intervened: {sorted(top_mlps)}")

        self.save_probe_results(top_mlps, probes, all_mlp_accs_np, all_mlp_f1s_np, all_mlp_kls_np, all_mlp_mses_np, self.probe_dir)
        self.logger.info(f"Probe consume time: {time.time()-probe_start}")
        self.probe_time = time.time()-probe_start

    def train_dnn(self, all_X_train, all_X_val, y_train, y_val, device, class_num=10):
        start_time = time.time()
        all_mlp_accs = []
        all_mlp_f1s = []
        all_mlp_kls = []
        all_mlp_mses = []
        all_mlp_cms = []
        probes = []
        for layer in tqdm(range(self.num_layers), desc="train_probes"): 
            X_train = all_X_train[:,layer,:]
            X_val = all_X_val[:,layer,:]
            self.logger.info(f'input_dim {X_train.shape[-1]}')
            model, loss_history = train_dnn(X_train, y_train, device=device, input_dim=X_train.shape[-1], class_num=class_num)
            self.logger.info(f"layer: {layer}, train_loss: {loss_history}")
            accuracy, f1, avg_kl_div, avg_mse, cm = evaluate_dnn(model, X_val, y_val, device=device, layer=layer)
            all_mlp_accs.append(accuracy)
            all_mlp_f1s.append(f1)
            all_mlp_kls.append(avg_kl_div)
            all_mlp_mses.append(avg_mse)
            probes.append(model)
            all_mlp_cms.append(cm)
        all_mlp_accs_np = np.array(all_mlp_accs)
        all_mlp_f1s_np = np.array(all_mlp_f1s)
        all_mlp_kls_np = np.array(all_mlp_kls)
        all_mlp_mses_np = np.array(all_mlp_mses)
        all_mlp_cms_np = all_mlp_cms
        self.logger.info(f"DNN training consume time: {time.time()-start_time}")
        return probes, all_mlp_accs_np, all_mlp_f1s_np, all_mlp_kls_np, all_mlp_mses_np, all_mlp_cms_np


    def save_probe_results(self, top_mlps, probes, all_mlp_accs_np, all_mlp_f1s_np, all_mlp_kls_np, all_mlp_mses_np, save_dir): 
        """takes in a list of pytorch dnn probes and saves them to path"""
        models_dict = {f"model_{i}": model.state_dict() for i, model in enumerate(probes)}
        
        torch.save(models_dict, os.path.join(save_dir, 'all_dnn_models.pth'))

        information_dict = {"all_mlp_accs_np": all_mlp_accs_np,
                    "all_mlp_f1s_np": all_mlp_f1s_np,
                    "all_mlp_kls_np": all_mlp_kls_np,
                    "all_mlp_mses_np": all_mlp_mses_np,
                    "top_mlps": top_mlps
                    }
        with open(os.path.join(save_dir, 'information_dict.pkl'), 'wb') as f:
            pickle.dump(information_dict, f)

    def load_probe_results(self, save_dir, device, input_dim=4096, class_num=3):
        """takes in a list of pytorch dnn probes and saves them to path"""
        models_dict = torch.load(os.path.join(save_dir,'all_dnn_models.pth'), map_location=device)
        probes = []
        for i in range(len(models_dict)): 
            model = DNN(input_dim=input_dim, class_num=class_num).to(device)
            model.load_state_dict(models_dict[f"model_{i}"])
            model = model.to(device)
            probes.append(model)
        information_dict = pickle.load(open(os.path.join(save_dir, 'information_dict.pkl'), 'rb'))
        top_mlps = information_dict["top_mlps"]
        all_mlp_accs_np = information_dict["all_mlp_accs_np"]
        all_mlp_f1s_np = information_dict["all_mlp_f1s_np"]
        all_mlp_kls_np = information_dict["all_mlp_kls_np"]
        all_mlp_mses_np = information_dict["all_mlp_mses_np"] 
        return top_mlps, probes, all_mlp_accs_np, all_mlp_f1s_np, all_mlp_kls_np, all_mlp_mses_np
    
    def slice_activations(self, act, last_indexs=[(0,1)]):
        activations = act.clone().detach()
        last_start, last_end = last_indexs[0]
        last_token = activations[:, last_start:last_end, :]
        return last_token, last_token.shape[1]


    def restore_activations(self, activations, sliced_activations, last_indexs=[(0,1)]):
        activations_restored = activations.clone().detach()
        last_start, last_end = last_indexs[0]
        activations_restored[:, last_start:last_end, :] = sliced_activations
        return activations_restored

    def get_interventions_dict_dnns_for_mlp(self, top_heads, probes, tuning_activations, noise_factor=3): 

        interventions = {}
        for layer in top_heads: 
            interventions[f"model.layers.{layer}.mlp"] = []

        stds = []
        l2_norms_list = []
        for layer in top_heads:
            dnn_model = probes[layer]
            activations = tuning_activations[:,layer,:] 
            activations = activations.astype(np.float64)
            l2_norms = np.linalg.norm(activations, axis=1)
            
            std = np.std(l2_norms)
            stds.append((layer, std))
            l2_norms_list.append((layer, l2_norms))
            noise_level = noise_factor * std
            interventions[f"model.layers.{layer}.mlp"].append((dnn_model, noise_level))
        
        stds = sorted(stds, key=lambda x: x[0], reverse=False)
        for layer, std in stds:
            self.logger.info(f"layer: {layer}, std: {std:.4f}")

        return interventions



    def run_debias(self):
        args = self.args
        debias_start = time.time()
        self.logger.info(f"begin to run debias")
        self.logger.info(f"model device: {self.model.device}")

        top_mlps, probes, all_mlp_accs_np, all_mlp_f1s_np, all_mlp_kls_np, all_mlp_mses_np =\
                self.load_probe_results(self.probe_dir, self.model.device,
                            input_dim=self.all_X_train.shape[-1], class_num=args.class_num)
        top_mlps = np.argsort(all_mlp_f1s_np)[::-1][:args.num_mlps]
        self.logger.info(f"interven layers: {sorted(top_mlps)}")
        interventions = self.get_interventions_dict_dnns_for_mlp(top_mlps, 
                            probes, self.all_X_train, noise_factor=args.noise_factor)
        def adv_remove(head_output, layer_name, last_indexs=[(0,1)]): 
            for dnn_model, noise_level in interventions[layer_name]:    
                sliced_activation, _ = self.slice_activations(head_output, last_indexs)
                device = next(dnn_model.parameters()).device
                adv_output, loss, use_iter = pgd_attack_general(model=dnn_model, embeddings=sliced_activation, epsilon=noise_level, \
                                            eps_iter=noise_level/args.eps_iter, num_iter=args.num_iter, device=device, loss_type=args.loss_type)
                self.logger.info(f"layer: {layer_name}, final adv loss: {loss:.4f}, use_iter: {use_iter}")

                sliced_activation = adv_output.to(dtype=head_output.dtype, device=head_output.device.index)

                head_output = self.restore_activations(head_output, sliced_activation, last_indexs)

            return head_output

        self.run_evaluation_on_task(self.model, self.tokenizer, 
                                    args.eval_task, args.test_file, 
                                    self.debias_dir, interventions, adv_remove)
        self.logger.info(f"Debias consume time: {time.time()-debias_start}")
        self.debias_time = time.time()-debias_start
    def run_evaluation_on_task(self, model, tokenizer, task, test_file, output_dir, interventions={}, intervention_fn=None):
        
        if task == "mmlu":
            evaluator = EvaluateMMLU(model, tokenizer, test_file, task, 
                                    interventions, intervention_fn)
        elif task == "bbq":
            evaluator = EvaluateBBQ(model, tokenizer, test_file, task, 
                                    interventions, intervention_fn,
                                    protect_attr=self.args.protect_attr,
                                    category=self.args.bbq_cate, debias_method="interven")
        elif task == "biasasker":
            evaluator = EvaluateBiasAsker(model, tokenizer, test_file, task, 
                                    interventions, intervention_fn,
                                    protect_attr=self.args.protect_attr,
                                    category=self.args.bbq_cate, debias_method="interven")

        else:
            raise ValueError(f"Unknown task {task}")

        evaluator: Evaluate
        evaluator.evaluate()
        evaluator.save_results(output_dir)


    def run_pipe(self):
        start_time = time.time()
        self.run_trace()
        self.load_dataset()
        self.run_probe()
        self.run_debias()
        end_time = time.time()
        self.logger.info(f"Total time taken: {end_time - start_time} seconds; {((end_time - start_time) / 60)} minutes")
        self.logger.info(f"Trace time: {self.trace_time} seconds; {self.trace_time / 60} minutes")
        self.logger.info(f"Probe time: {self.probe_time} seconds; {self.probe_time / 60} minutes")
        self.logger.info(f"Debias time: {self.debias_time} seconds; {self.debias_time / 60} minutes")

if __name__ == "__main__":
    args = args_parser()
    start_time = time.time()
    adv = Ours(args)
    adv.run_pipe()
    end_time = time.time()
    adv.logger.info(f"Main program Total time taken: {end_time - start_time} seconds; {((end_time - start_time) / 60)} minutes")
