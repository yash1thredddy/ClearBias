RESULTS_DIR = "../results"

HF_NAMES = {
    'llama2_chat_7B': '/mnt/data/ssd/yisong/llm_data/meta-llama/Llama-2-7b-chat-hf',
    'llama2_chat_13B': '/mnt/data/ssd/yisong/llm_data/meta-llama/Llama-2-13b-chat-hf',
    'llama3_8B_instruct': '/mnt/data/ssd/yisong/llm_data/meta-llama/Meta-Llama-3-8B-Instruct',
    'llama3_1_8B_instruct': '/mnt/data/ssd/yisong/llm_data/meta-llama/Meta-Llama-3.1-8B-Instruct',
}

from matplotlib import pyplot as plt
def plot_line(data, name='test.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(data, marker='o', linestyle='-', color='b')
    plt.title(f'{name.split("/")[-1].split(".")[0]}')
    plt.xlabel('Layer Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig(name)
