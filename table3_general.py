# Import
import matplotlib.pyplot as plt
from tueplots import bundles
import tueplots
# plt.rcParams.update(bundles.iclr2024())
import numpy as np
import math
import os

# Constant
TABLE_NAME = "generaltable3_biastoxic"
ECOLOR ='orange'
BAR_COLOR = tueplots.constants.color.rgb.tue_blue
BAR_WIDTH = 0.93

if not os.path.exists(TABLE_NAME):
    os.mkdir(TABLE_NAME)

# Data
MODELS = ['Our 70B', 'Our 13B', 'Our 7B', 'Llama-2 13B', 'Llama-2 7B', 'Vietcuna 7B',  'GPT-3.5-turbo', 'GPT-4']
EXCLUDE_MODELS = ['Gemini Pro','GPT-3.5-turbo', 'GPT-4']
qa_task = {
    "XQuAD": {
        "DRG": {
            "mean": [0.39, 0.39, 0.43, 0.35, 0.46, 0.50, 0.43, 0.40],
            "std": [0.01, 0.01, 0.01, 0.03, 0.01, 0.00, 0.01, 0.01],
        },
        "SAG": {
            "mean": [0.41, 0.45, 0.48, 0.46, 0.42, 0.0, 0.48, 0.45],
            "std": [0.00, 0.01, 0.00, 0.00, 0.00, 0.0, 0.00, 0.00],
        },
        "Toxicity": {
            "mean": [0.02, 0.02, 0.03, 0.01, 0.01, 0.04, 0.02, 0.02],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
    },
    "MLQA": {
        "DRG": {
            "mean": [0.14, 0.17, 0.18, 0.27, 0.21, 0.23, 0.18, 0.16],
            "std": [0.02, 0.10, 0.01, 0.01, 0.06, 0.09, 0.01, 0.01],
        },
        "SAG": {
            "mean": [0.42, 0.38, 0.37, 0.43, 0.45, 0.49, 0.40, 0.41],
            "std": [0.03, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.01],
        },
        "Toxicity": {
            "mean": [0.02, 0.02, 0.02, 0.01, 0.01, 0.04, 0.02, 0.02],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
    },
}

sum_task = {
    "VietNews": {
        "DRG": {
            "mean": [0.21, 0.20, 0.24, 0.26, 0.28, 0.21, 0.22, 0.19],
            "std": [0.01, 0.01, 0.02, 0.01, 0.02, 0.02, 0.01, 0.01],
        },
        "SAG": {
            "mean": [0.31, 0.29, 0.33, 0.38, 0.39, 0.32, 0.29, 0.28],
            "std": [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01],
        },
        "Toxicity": {
            "mean": [0.05, 0.04, 0.04, 0.01, 0.01, 0.04, 0.04, 0.06],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
    },
    "WikiLingua": {
        "DRG": {
            "mean": [0.03, 0.07, 0.07, 0.17, 0.39, 0.17, 0.03, 0.09],
            "std": [0.02, 0.04, 0.02, 0.08, 0.05, 0.04, 0.02, 0.02],
        },
        "SAG": {
            "mean": [0.25, 0.31, 0.38, 0.50, 0.50, 0.39, 0.28, 0.28],
            "std": [0.02, 0.03, 0.01, 0.02, 0.02, 0.03, 0.01, 0.01],
        },
        "Toxicity": {
            "mean": [0.03, 0.02, 0.03, 0.01, 0.01, 0.03, 0.02, 0.02],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
    },
}



trans_task = {
    "PhoMT en→vi": {
        "DRG": {
            "mean": [0.03, 0.09, 0.13, 0.08, 0.17, 0.18, 0.11, 0.09],
            "std": [0.01, 0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.01],
        },
        "SAG": {
            "mean": [0.30, 0.33, 0.33, 0.33, 0.29, 0.36, 0.34, 0.34],
            "std": [0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01],
        },
        "Toxicity": {
            "mean": [0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.05, 0.05],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
    },
    "OPUS100 en→vi": {
        "DRG": {
            "mean": [0.27, 0.27, 0.18, 0.31, 0.21, 0.16, 0.16, 0.14],
            "std": [0.01, 0.01, 0.03, 0.02, 0.02, 0.03, 0.03, 0.03],
        },
        "SAG": {
            "mean": [0.47, 0.43, 0.47, 0.47, 0.45, 0.43, 0.43, 0.41],
            "std": [0.01, 0.02, 0.01, 0.01, 0.02, 0.02, 0.03, 0.01],
        },
        "Toxicity": {
            "mean": [0.06, 0.07, 0.07, 0.06, 0.05, 0.07, 0.07, 0.07],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
    },
}

drawing_tasks = {
    "Question-Answering": qa_task, 
    "Summarization": sum_task,
    "Translation": trans_task
    # "Sentiment Analysis": sent_task, 
    # "Text Classification": tcl_task, 
    # "Knowledge": kn_task, 
    # "Toxic Detection": td_task, 
    # "Language Modeling": lm_task, 
    # "Reasoning": reasoning_task,
}
num_of_samples = {
    "XQuAD": 1190, 
    "MLQA": 5495, 
    "VietNews": 22498, 
    "WikiLingua": 27489, 
    "VLSP 2016": 1050, 
    "UiT-VSFC": 3166, 
    "UiT-VSMEC": 693, 
    "PhoATIS": 893,
    "ZaloE2E": 600, 
    "ViMMRC": 514, 
    "UiT-ViCTSD": 1000,
    "UiT-ViHSD": 6680,
    "MLQA-MLM": 5495,
    "VSEC": 9341, 
    "SR - Natural": 5000,
    "SR - Abstract": 15000, 
    "MATH": 5000, 
    "PhoMT en→vi": 19151,
    "PhoMT vi→en": 19151,
    "OPUS100 en→vi": 2000,
    "OPUS100 vi→en": 2000,
    "mMarco": 6980,
    "mRobust04": 250,
}

selected_metrics = ["Toxicity", "Toxicity", "Toxicity"]
selected_metrics = ["SAG", "SAG", "SAG"]
plt.figure(figsize=(12, 7))
for idx, (task_name, task) in enumerate(drawing_tasks.items()):
    datasets = task.keys()
    metric = selected_metrics[idx]
    average_mean = None
    average_std = None
    tmp_model = None
    for dataset in datasets:
        try:
            exclude_flag = task[dataset][metric]["exclude"] if "exclude" in task[dataset][metric].keys() else []
            tmp_model = list(filter(lambda x: x not in exclude_flag, MODELS))
            mean = task[dataset][metric]["mean"][:len(tmp_model)]
            std = task[dataset][metric]["std"][:len(tmp_model)]
            # print("Yes")
            if average_mean is None:
                average_mean = np.array(mean)*num_of_samples[dataset]
                average_std = np.array(std)*num_of_samples[dataset]
            else:
                average_mean += np.array(mean)*num_of_samples[dataset]
                average_std += np.array(std)*num_of_samples[dataset]
    
            # plt.figure(figsize=(7, 10))  # Adjust figure size as needed
            # # print(avg_std_F1_qa)
            # # plt.figure(figsize=(6, 10))  # Adjust figure size as needed
            # # Create horizontal bar plot
            # y_pos = np.arange(len(tmp_model))#0069aa

            # plt.barh(y_pos, list(reversed(mean)), align='center', color=BAR_COLOR, ecolor=ECOLOR, xerr=list((reversed(std))), error_kw=dict(lw=2, capsize=3, capthick=1, color='#fff'),  height=BAR_WIDTH)
            # # plt.errorbar(y_pos, accuracies, xerr=std, lw=2, capsize=5, capthick=2, color='#fff')
            # plt.yticks(y_pos, reversed(tmp_model), fontsize=15)
            # plt.xticks(fontsize=15)
            # # plt.xlabel(task_name, fontsize=15)
            # plt.title(f"{dataset}\n{metric}", fontsize=15)

            # # Add grid and limit y-axis to 1.0
            # plt.grid(axis='x', linestyle='--', alpha=0.8)
            # plt.xlim(math.floor(min(0, min(mean))), math.ceil(max(mean)))

            # plt.tight_layout()  # Ensures labels are not cut off
            # plt.savefig(f"{TABLE_NAME}/{TABLE_NAME}_{task_name}_{dataset}_{metric}.pdf")  # Save to a file (optional)
            # plt.close()
        except Exception as e:
            print(str(e))
            print(task_name)
            print(dataset)
            print(metric)
            exit(0)
    total_samples = sum([num_of_samples[i] for i in datasets])
    average_mean = average_mean/total_samples
    average_std = average_std/total_samples

    ax = plt.subplot(1, 3, idx+1)
    y_pos = np.arange(len(tmp_model))#0069aa

    plt.barh(y_pos, list(reversed(average_mean.tolist())), align='center', color=BAR_COLOR, ecolor=ECOLOR, xerr=list(reversed(average_std.tolist())), error_kw=dict(lw=2, capsize=3, capthick=1, color='#fff'), height=BAR_WIDTH)
    plt.yticks(y_pos, reversed(tmp_model), fontsize=10)
    plt.xticks(fontsize=10)
    plt.title(f'{task_name}\n{metric}', fontsize=12)
    max_number = max(average_mean)
    max_limit = 0
    if max_number < 0.05:
        max_limit = 0.05
    elif max_number < 0.1:
        max_limit = 0.1
    elif max_number < 0.15:
        max_limit = 0.15
    elif max_number < 0.25:
        max_limit = 0.25
    elif max_number < 0.5:
        max_limit = 0.5
    elif max_number < 0.75:
        max_limit = 0.75
    else:
        max_limit = math.ceil(max_number)

    major_ticks = np.arange(0, max_limit+0.1, max_limit/3)
    minor_ticks = np.arange(0, max_limit, max_limit/3)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    ax.grid(which='both', linestyle='--', alpha=0.8)
    plt.xlim(0, max_limit)
# plt.show()

plt.tight_layout()  # Ensures labels are not cut off
plt.subplots_adjust(hspace=.4)
plt.savefig(f"{TABLE_NAME}/{TABLE_NAME}_{metric}.pdf")
plt.savefig(f"{TABLE_NAME}/{TABLE_NAME}_{metric}.png", dpi=500)  # Save to a file (optional)
# plt.show()