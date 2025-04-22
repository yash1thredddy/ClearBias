import os
import json

import numpy as np
import torch
from tqdm import tqdm

from evaluation.evaluate import Evaluate
from utils.globals import *
from baukit import Trace, TraceDict

import logging
logger = logging.getLogger(name='logger')

def get_llama_activations_bau(model, tok, prompt): 
    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    input_ids = tok.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        with TraceDict(model, HEADS+MLPS) as ret:
            output = model(input_ids, output_hidden_states = True)

        logits = output.logits.float()
        probabilities = logits.softmax(dim=2)[:,-1,:].squeeze()

        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
        hidden_states = hidden_states.detach().cpu().numpy()
        head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
        mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
        mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()

    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states, probabilities.tolist()

def preprocess(text:str, sentence_suffix:str):
    if text.count("[") != 2:
        return None, None
    bias_concept = text.split("[")[1].split("]")[0]
    social_group = text.split("[")[2].split("]")[0]
    clean_sentence = text.replace(f"[{bias_concept}]", bias_concept.lower()).split(f"[")[0].strip()
    return clean_sentence, bias_concept.lower()

class EvaluateTrace(Evaluate):

    def __init__(self, model, tok, test_file, task, protect_attr='gender'):
        super().__init__(model, tok, test_file, task)

        assert self.task == "trace", f"Task class mismatch:, expected 'trace', got '{self.task}' instead"
        self.protect_attr = protect_attr
        self.load_attr(protect_attr)

        self.results = {}
        self.partial_results = {}
        for item in self.attr_words:
            self.results[f"prob_{item}"] = 0.
            self.results[f"predicted_{item}"] = 0. 
            self.partial_results[f"prob_{item}"] = []
        self.partial_results[f"predicted_token"] = []
        self.partial_results["prompt"] = []

        self.load_data()

    
    def load_attr(self, protect_attr):
        def load_attr_words(protect_attr, _label):
            attr_words_base_dir = "../biased_knowledge"
            if protect_attr in ['Age', 'Disability_status', 'Gender_identity', 
                            'Nationality', 'Physical_appearance', 
                            'Race_ethnicity', 'SES', 
                            'Sexual_orientation', 'Religion']:
                file = os.path.join(attr_words_base_dir, protect_attr, "group.json")
                with open(file, 'r') as f:
                    data = json.load(f)
                    attr_words = sorted(list(data.keys()))
                    return attr_words, data
            else:
                sub_dir = os.path.join(attr_words_base_dir, protect_attr, _label+".txt")
                with open(sub_dir, 'r') as f:
                    lines = f.readlines()
                    word_list = [line.strip() for line in lines if line.strip() != ""]
                    word_list = list(set(word_list))
                    f.close()
                    return word_list

        self.attr_words = []
        self.attr_words_map = {}

        if protect_attr in ['Age', 'Disability_status', 'Gender_identity', 
                            'Nationality', 'Physical_appearance', 
                            'Race_ethnicity', 'SES', 
                            'Sexual_orientation', 'Religion']:
            self.attr_words, self.attr_words_map = load_attr_words(protect_attr, None)
            self.sentence_suffix = f" [{protect_attr}] group."
        

        self.attr_words_tokens_map = {}
        for item in self.attr_words:
            self.attr_words_tokens_map[item] = [self.tok.encode(token)[0] for token in self.attr_words_map[item]]
            self.attr_words_tokens_map[item] = list(set(self.attr_words_tokens_map[item]))
        logger.info(f"attr_words: {self.attr_words}")
        logger.info(f"attr_words_map: {self.attr_words_map}")
        logger.info(f"attr_words_tokens_map: {self.attr_words_tokens_map}")
        
    def load_data(self):
        self.generation_prompts = None
        with open(self.test_file, 'r') as f:
            self.generation_prompts = json.load(f)
        
        
    def slice_activations(self, activations):
        last_token = activations[:, -1, :].copy()
        return [last_token], 1
        
    
    def evaluate(self):
        
        self.all_layer_wise_activations = []
        self.all_head_wise_activations = []
        self.all_mlp_wise_activations = []
        self.sample_arrays = []
        self.all_probabilities = []

        total_trace = 0
        for _target, generated_sentences in tqdm(self.generation_prompts.items(), desc="Evaluating generation prompts"):
            sentences = generated_sentences
            for raw_sentence in sentences:
                sentence, bias_concept = preprocess(raw_sentence, self.sentence_suffix)
                if sentence is None:
                    continue
                if bias_concept != _target:
                    continue
                total_trace += 1
                logger.info(f"sentence: {sentence}, bias_concept: {bias_concept}")
                
                layer_wise_activations, head_wise_activations, mlp_wise_activations, probabilities = get_llama_activations_bau(self.model, self.tok, sentence)
            
                layer_act, _len = self.slice_activations(layer_wise_activations)
                self.all_layer_wise_activations.extend(layer_act)
                head_act, _len = self.slice_activations(head_wise_activations)
                self.all_head_wise_activations.extend(head_act)
                mlp_act, _len = self.slice_activations(mlp_wise_activations)
                if len(mlp_act) != _len:
                    logger.info(f"{sentence} fail to capture {bias_concept}")
                    _len = len(mlp_act)
                self.all_mlp_wise_activations.extend(mlp_act)
                self.sample_arrays.append(_len)
        
                predicted_token = self.tok.decode([probabilities.index(max(probabilities))])
                self.partial_results["predicted_token"].append(predicted_token)
                for item in self.attr_words:
                    attr_probs = [probabilities[token_id] for token_id in self.attr_words_tokens_map[item]]
                    self.partial_results[f"prob_{item}"].append(sum(attr_probs))
                self.partial_results["prompt"].append(sentence)

                self.all_probabilities.append(probabilities)
        

        logger.info(f"Total traces sentences number: {total_trace}")
        for item in self.attr_words:
            self.results[f"prob_{item}"] = np.mean(self.partial_results[f"prob_{item}"])

            sum_item = 0
            for i in range(len(self.attr_words_map[item])):
                sum_item += self.partial_results["predicted_token"].count(self.attr_words_map[item][i])
            self.results[f"predicted_{item}"] = sum_item / len(self.generation_prompts)

        _all_partial = []
        for item in self.attr_words:
            _all_partial.append(self.partial_results[f"prob_{item}"])
        _all_partial = np.column_stack(_all_partial)
        row_diff = np.max(_all_partial, axis=1) - np.min(_all_partial, axis=1)
        self.partial_results["diff_prob"] = row_diff
        self.results["diff_prob"] = np.mean(row_diff)

        self.partial_results = [dict(zip(self.partial_results, t)) for t in zip(*self.partial_results.values())]
      

    def save_results(self, result_dir):
        test_name = os.path.basename(self.test_file).split(".")[0]
        with open(os.path.join(result_dir, f"res_{self.task}_{test_name}.json"), 'w') as f:
            json.dump(self.results, f, indent=4)

        with open(os.path.join(result_dir, f"partial_res_{self.task}_{test_name}.json"), 'w') as f:
            json.dump(self.partial_results, f, indent=4)

        logger.info("Saving layer wise activations")
        _name = os.path.join(result_dir, f"features_{self.task}_{test_name}_layer_wise.npy")
        np.save(_name, self.all_layer_wise_activations)

        logger.info("Saving head wise activations")
        _name = os.path.join(result_dir, f"features_{self.task}_{test_name}_head_wise.npy")
        np.save(_name, self.all_head_wise_activations)

        logger.info("Saving mlp wise activations")
        _name = os.path.join(result_dir, f"features_{self.task}_{test_name}_mlp_wise.npy")
        np.save(_name, self.all_mlp_wise_activations)

        logger.info("Saving sample arrays")
        _name = os.path.join(result_dir, f"sample_arrays_{self.task}_{test_name}.npy")
        np.save(_name, self.sample_arrays)

        logger.info("Saving probabilities")
        _name = os.path.join(result_dir, f"probabilities_{self.task}_{test_name}.npy")
        np.save(_name, self.all_probabilities)