import os
import sys

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from IPython.display import display
from matplotlib.patches import Patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
script_dir = os.path.dirname(os.path.abspath(__file__))

# Font settings
FONT_SMALL = 16
FONT_MEDIUM = 18
FONT_LARGE = 20

plt.rcParams['font.size'] = FONT_LARGE
plt.rcParams['axes.labelsize'] = FONT_MEDIUM
plt.rcParams['xtick.labelsize'] = FONT_SMALL
plt.rcParams['ytick.labelsize'] = FONT_SMALL
plt.rcParams['legend.fontsize'] = FONT_SMALL
plt.rcParams['figure.titlesize'] = FONT_MEDIUM
plt.rcParams['axes.titlesize'] = FONT_MEDIUM
plt.rcParams['figure.figsize'] = (16, 8)


def load_results_data():
    order = ["base-pretrained", "base-confidence", "base-uncertainty", "ensemble", "mc-dropout", "edl-meh"]
    
    path_base = os.path.join(script_dir, '..', 'interim_results', 'detect')
    newest_results = 'data_splits_and_models'
    print(f'Results folder: {newest_results}')
    
    datasets = {}
    train_dataset_path = [f for f in os.listdir(f'{path_base}/{newest_results}') 
                          if f.startswith('train') and os.path.isdir(f'{path_base}/{newest_results}/{f}')][0]
    print(f'Train dataset path: {train_dataset_path}')
    
    for dataset_type in ['val']:
        dataset_paths = [f for f in os.listdir(f'{path_base}/{newest_results}') 
                        if f.startswith(dataset_type) and os.path.isdir(f'{path_base}/{newest_results}/{f}')]
        
        for dataset_path in dataset_paths:
            full_dataset_path = f'{path_base}/{newest_results}/{dataset_path}'
            df_results = pd.DataFrame()
            
            for model_folder in [f for f in os.listdir(full_dataset_path) 
                                if os.path.isdir(f'{full_dataset_path}/{f}')]:
                df_results_model = pd.read_csv(f'{full_dataset_path}/{model_folder}/results_extended.csv')
                
                if df_results.empty:
                    df_results = df_results_model
                else:
                    df_results = pd.concat([df_results, df_results_model])
            
            df_results.set_index('name', inplace=True)
            df_results = df_results.loc[order, :]
            
            print(f'Processing dataset: {dataset_path}')
            display(df_results)
            
            speed_cols = [col for col in df_results.columns if 'speed' in col]
            df_results['speed_total'] = df_results[speed_cols].sum(axis=1)
            
            datasets[dataset_path] = df_results
    
    return datasets, train_dataset_path


def format_model_names(datasets):
    names_title_case = ['Base Pretrained', 'Base Confidence', 'Base Uncertainty', 
                       'Ensemble', 'MC Dropout', 'EDL MEH']
    
    rename_dict_index = {
        'base-pretrained': 'Base Pretrained',
        'base-confidence': 'Base Confidence', 
        'base-uncertainty': 'Base Uncertainty',
        'ensemble': 'Ensemble',
        'mc-dropout': 'MC Dropout',
        'edl-meh': 'EDL MEH'
    }
    
    for df in datasets.values():
        df.rename(index=rename_dict_index, inplace=True)
    
    return datasets, names_title_case


def create_dual_barplots(datasets, train_dataset_path):
    metric1 = 'metrics/mAP50(B)'
    metric2 = 'metrics/mUE50'
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()
    
    for idx, (key, df) in enumerate(datasets.items()):
        if idx >= 4:
            break
        ax1 = axs[idx]
        
        color_1 = 'royalblue'
        ax1.set_xlabel('Model Name')
        ax1.set_ylabel('mAP50', color=color_1)
        ax1.bar(df.index, df[metric1], color=color_1, width=0.4, label=metric1)
        ax1.tick_params(axis='y', labelcolor=color_1)
        
        ax2 = ax1.twinx()
        color_2 = 'orangered'
        ax2.set_ylabel('mUE50', color=color_2)
        ax2.bar([i + 0.4 for i in range(len(df.index))], df[metric2], color=color_2, width=0.4, label=metric2)
        ax2.tick_params(axis='y', labelcolor=color_2)
        
        ax1.set_xticks([i + 0.2 for i in range(len(df.index))])
        ax1.set_xticklabels(df.index, rotation=45, ha='right')
        ax1.set_title(f"{key.replace('val-', '').replace('-from-coco80', '').replace('-coco80', '').replace('-', ' ').title().replace('Raincityscapes', 'RainCityscapes')} Validation")
        
        ax1.set_ylim(0, 0.6)
        ax2.set_ylim(0, 0.6)
    
    fig.tight_layout()
    fig.suptitle(f"Results for YOLO11n-based Models trained on {train_dataset_path.split('train')[1].replace('-from-coco80', '').replace('-coco80', '').replace('-', '').title()} Dataset", fontsize=FONT_LARGE)
    plt.subplots_adjust(top=0.92)
    plt.show()


def create_extended_barplots(datasets, train_dataset_path):

    metrics_config = {  # metrics to plot
        'mAP50': {
            'column': 'metrics/mAP50(B)',
            'ylabel': 'mAP50',
            'color': 'royalblue',
            'ylim': (0, 0.55),
            'higher_better': True
        },
        'Precision': {
            'column': 'metrics/precision(B)',
            'ylabel': 'Precision',
            'color': 'mediumseagreen',
            'ylim': (0, 1.0),
            'higher_better': True
        },
        'Recall': {
            'column': 'metrics/recall(B)',
            'ylabel': 'Recall',
            'color': 'darkorange',
            'ylim': (0, 0.55),
            'higher_better': True
        },
        'FPS': {
            'column': 'speed_total',
            'ylabel': 'FPS (@A100)',
            'color': 'mediumpurple',
            'ylim': (0, None),
            'higher_better': True,
            'transform': lambda x: 1000/x
        },
        'mUE50': {
            'column': 'metrics/mUE50',
            'ylabel': 'mUE',
            'color': 'orangered',
            'ylim': (0, 0.55),
            'higher_better': False
        },
        'AUROC50': {
            'column': 'metrics/AUROC50',
            'ylabel': 'AUROC',
            'color': 'forestgreen',
            'ylim': (0.5, 1.0),
            'higher_better': True
        },
        'FPR95_50': {
            'column': 'metrics/FPR95_50',
            'ylabel': 'FPR95',
            'color': 'crimson',
            'ylim': (0, 1.0),
            'higher_better': False
        }
    }
    
    num_datasets = len(datasets)
    fig, axs = plt.subplots(len(metrics_config), num_datasets, figsize=(4 * num_datasets, 4 * len(metrics_config)))
    
    if num_datasets == 1:
        axs = axs.reshape(-1, 1)
    
    for metric_idx, (metric_name, config) in enumerate(metrics_config.items()):
        for dataset_idx, (dataset_name, df) in enumerate(datasets.items()):
            ax = axs[metric_idx, dataset_idx]
            
            if config['column'] in df.columns:
                values = df[config['column']]
                                
                # Apply transformation if specified (e.g., 1/x for FPS)
                if 'transform' in config:
                    values = values.apply(config['transform'])
                
                bars = ax.bar(df.index, values, color='grey')
                
                # Handle dynamic ylim for FPS
                if config['ylim'][1] is None:
                    ax.set_ylim(0, values.max() * 1.1)
                else:
                    ax.set_ylim(config['ylim'])
                    
                ax.set_ylabel(config['ylabel'])
                ax.set_title(f"{dataset_name.replace('val-', '').replace('-from-coco80', '').replace('-coco80', '').replace('-', ' ').title().replace('Raincityscapes', 'RainCityscapes')}")
                ax.tick_params(axis='x', rotation=45)
                ax.set_xticks(range(len(df.index)))
                ax.set_xticklabels(df.index, rotation=45, ha='right')
                
                # Color code bars based on performance (green for best, red for worst)
                if config['higher_better']:
                    best_idx = np.argmax(values)
                    worst_idx = np.argmin(values)
                else:
                    best_idx = np.argmin(values)
                    worst_idx = np.argmax(values)
                
                bars[best_idx].set_color('lightgreen')
                bars[worst_idx].set_color('lightcoral')
            else:
                ax.text(0.5, 0.5, f'{config["column"]}\nnot available', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=FONT_MEDIUM)
                ax.set_title(f"{dataset_name.replace('val-', '').replace('-from-coco80', '').replace('-coco80', '').title()}")
    
    legend_elements = [
        Patch(facecolor='lightgreen', label='Best'),
        Patch(facecolor='lightcoral', label='Worst'),
        Patch(facecolor='grey', label='Others')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.93), 
               ncol=3, fontsize=FONT_SMALL)
    
    plt.tight_layout()
    plt.suptitle(f"Metrics for YOLO11n-based Models trained on {train_dataset_path.split('train')[1].replace('-', '').replace('fromcoco80', '').replace('coco80', '').title()} Dataset", 
                fontsize=FONT_LARGE, y=0.94)
    plt.subplots_adjust(top=0.90)
    plt.show()


def create_radar_chart(datasets, train_dataset_path, chosen_val_dataset='val-bdd100k-coco80'):
    if chosen_val_dataset not in datasets:
        chosen_val_dataset = list(datasets.keys())[0]
        print(f"Dataset '{chosen_val_dataset}' not found, using {chosen_val_dataset}")
    
    df_selected = datasets[chosen_val_dataset]
    
    rename_dict_cols = {
        'metrics/precision(B)': 'Precision',
        'metrics/recall(B)': 'Recall',
        'metrics/mAP50(B)': 'mAP50',
        'speed_total': 'FPS (@A100)',
        'metrics/mUE50': 'mUE50',
    }
    
    color_list = ['k', 'tomato', 'deepskyblue', 'limegreen', 'gold', 'orchid', 'lightcoral', 'lightseagreen']
    colors = {idx: color for idx, color in zip(df_selected.index, color_list[:len(df_selected.index)])}
    
    df = df_selected[rename_dict_cols.keys()].copy()
    # Convert ms to fps (1000ms/s divided by ms per image = fps)
    df.loc[:, 'speed_total'] = 1000 / df['speed_total']
    # Invert mUE so lower values (better) become higher values for radar visualization
    df.loc[:, 'metrics/mUE50'] = 1 / df['metrics/mUE50']
    # Take base index, and normalize all values to it
    df = df / df.iloc[0]
    
    df.rename(columns=rename_dict_cols, inplace=True)
    
    categories = list(df.columns)
    num_vars = len(categories)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Repeat the first angle to close the plot
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for idx in df.index:
        values = df.loc[idx].tolist()
        values += values[:1]  # Repeat the first value to close the plot
        ax.fill(angles, values, alpha=0.50, color=colors[idx], label=idx)
        ax.plot(angles, values, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    ylim = round(max(df.max()), 1) + 0.1
    ax.set_ylim(0, ylim)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    ax.tick_params(axis='x', pad=20)
    ax.set_yticks(np.arange(0, ylim, 0.2))
    ax.set_yticklabels(ax.get_yticks(), rotation=45)
    
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
    ax.set_theta_offset(np.pi / 2)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(1, color='black', linewidth=1, linestyle='--')
    ax.set_rlabel_position(-45)
    dataset_name = chosen_val_dataset.replace('val-', '').replace('-from-coco80', '').replace('-coco80', '').replace('-', ' ').title().replace('Raincityscapes', 'RainCityscapes')
    ax.set_title(f"{dataset_name} Validation Performance relative to base model (100%)".title(), pad=60)
    ax.title.set_position([.5, 1.4])
    
    # Save the plot
    path_base = os.path.join(script_dir, '..', 'interim_results', 'detect')
    newest_results = 'data_splits_and_models'
    path = f'{path_base}/{newest_results}/{chosen_val_dataset}'
    plt.savefig(f'{path}/radar_chart.png', bbox_inches='tight', dpi=600)
    plt.savefig(f'{path}/radar_chart.pdf', bbox_inches='tight')
    plt.show()