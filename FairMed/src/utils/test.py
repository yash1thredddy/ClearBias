

import numpy as np


mlp_wise = "/mnt/data/ssd/yisong/llm_debias/code_output/FairMed/results/AdvDebias/llama2_chat_7B/BBQ/Age/exp1_noise_factor_3/trace/features_trace_Age_sentences_mlp_wise.npy"
mlp_wise_activations = np.load(mlp_wise)

mlp_wise = "/mnt/data/ssd/yisong/llm_debias/adv/results/AdvDebias/llama2_chat_7B/BBQ_speed/Age/last_token_v2_noR_subjectN_bestloss_speed/trace/features_gen_trace_Age_sentences_v2_mlp_wise.npy"
mlp_wise_activations_2 = np.load(mlp_wise)

for i in range(32):
    for j in range(0, mlp_wise_activations.shape[0]):
        print(mlp_wise_activations[j,i,:])
        print(mlp_wise_activations_2[j,i,:])
        if mlp_wise_activations[j,i,:]!=mlp_wise_activations_2[j,i,:]:
            print("Different")
            print(i, j)
            print(mlp_wise_activations[j,i,:])
            print(mlp_wise_activations_2[j,i,:])
            break