from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from dataclasses import dataclass
from typing import Callable, Optional

FONT_SMALL = 16
FONT_MEDIUM = 18
FONT_LARGE = 20
plt.rcParams.update({
    'font.size': FONT_LARGE,
    'axes.labelsize': FONT_MEDIUM,
    'xtick.labelsize': FONT_SMALL,
    'ytick.labelsize': FONT_SMALL,
    'legend.fontsize': FONT_SMALL,
    'figure.titlesize': FONT_MEDIUM,
    'axes.titlesize': FONT_MEDIUM,
    'figure.figsize': (16, 8),
})

MODEL_ORDER_PRETTY = [
    'Base Pretrained', 'Base Confidence', 'Base Uncertainty',
    'Ensemble', 'MC Dropout', 'EDL MEH'
]

def order_models(idx_like):
    """Return models in fixed order, dropping any missing."""
    present = set(idx_like)
    return [m for m in MODEL_ORDER_PRETTY if m in present]

VAL_ORDER_RAW = [
    'cityscapes-from-coco80', 'foggy-cityscapes-from-coco80',
    'raincityscapes-from-coco80', 'kitti-from-coco80',
    'bdd100k-coco80', 'nuimages-coco80'
]
def order_vals(vals):
    vals = list(vals)
    ordered = [v for v in VAL_ORDER_RAW if v in vals]
    tail = [v for v in vals if v not in VAL_ORDER_RAW]
    return ordered + tail

MODEL_ORDER_RAW = ["base-pretrained", "base-confidence", "base-uncertainty",
                   "ensemble", "mc-dropout", "edl-meh"]
MODEL_RENAME = {
    'base-pretrained': 'Base Pretrained',
    'base-confidence': 'Base Confidence',
    'base-uncertainty': 'Base Uncertainty',
    'ensemble': 'Ensemble',
    'mc-dropout': 'MC Dropout',
    'edl-meh': 'EDL MEH',
}

@dataclass
class MetricCfg:
    column: str
    ylabel: str
    ylim: tuple
    higher_better: bool
    transform: Optional[Callable[[pd.Series], pd.Series]] = None

METRICS = {
    'mAP50': MetricCfg('metrics/mAP50(B)', 'mAP50 →', (0, 0.55), True),
    'Precision': MetricCfg('metrics/precision(B)', 'Precision →', (0, 1.0), True),
    'Recall': MetricCfg('metrics/recall(B)', 'Recall →', (0, 0.55), True),
    'FPS': MetricCfg('speed_total', 'FPS (@A100) →', (0, None), True, transform=lambda s: 1000.0 / s),  # ms → FPS
    'mUE50': MetricCfg('metrics/mUE50', '← mUE', (0, 0.55), False),
    'AUROC50': MetricCfg('metrics/AUROC50', 'AUROC →', (0.5, 1.0), True),
    'FPR95_50': MetricCfg('metrics/FPR95_50', '← FPR95', (0, 1.0), False),
    'E-AURC50': MetricCfg('metrics/E-AURC50', '← E-AURC', (0, 0.12), False),
}

def format_dataset_name(name):
    return (name.replace('-from-coco80', '')
                .replace('-coco80', '')
                .replace('-', ' ')
                .title()
                .replace('Raincityscapes', 'RainCityscapes'))

def _sum_speed_cols(df):
    speed_cols = [c for c in df.columns if 'speed' in c if not 'loss' in c] # inference speed only
    return df[speed_cols].sum(axis=1) if speed_cols else pd.Series([np.nan]*len(df), index=df.index)

def load_all_results(results_root):
    """
    returns a tidy df with MultiIndex (train,val,model) and result columns,
    including computed 'speed_total'.
    """
    rows = []
    for train_dir in sorted([p for p in results_root.iterdir() if p.is_dir() and p.name.startswith('train-')]):
        train = train_dir.name.replace('train-', '')
        for val_dir in sorted([p for p in train_dir.iterdir() if p.is_dir() and p.name.startswith('val-')]):
            val = val_dir.name.replace('val-', '')
            for model_dir in sorted([p for p in val_dir.iterdir() if p.is_dir()]):
                csv_path = model_dir / 'results_extended.csv'
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path)
                if 'name' not in df.columns:
                    continue
                df = df.set_index('name')
                available = [m for m in MODEL_ORDER_RAW if m in df.index]
                if not available:
                    continue
                df = df.loc[available].copy()
                df['speed_total'] = _sum_speed_cols(df)
                df['__train'] = train
                df['__val'] = val
                df['__model'] = df.index
                rows.append(df)

    if not rows:
        return pd.DataFrame()

    df_all = pd.concat(rows, axis=0, ignore_index=False)
    df_all['model'] = df_all['__model'].map(MODEL_RENAME).fillna(df_all['__model'])
    idx = pd.MultiIndex.from_arrays([df_all['__train'], df_all['__val'], df_all['model']],
                                    names=['train', 'val', 'model'])
    df_all = df_all.drop(columns=['__train','__val','__model'])
    df_all = df_all.set_index(idx)
    return df_all

def color_best_worst(bars, values, higher_better):
    if len(values) == 0:
        return
    best = np.argmax(values) if higher_better else np.argmin(values)
    worst = np.argmin(values) if higher_better else np.argmax(values)
    bars[best].set_color('lightgreen')
    bars[worst].set_color('lightcoral')

def setup_axis(ax, names, values, cfg, title):
    ymin, ymax = cfg.ylim
    if ymax is None:
        ymax = (max(values)*1.1) if len(values) else 1.0
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(cfg.ylabel)
    ax.set_title(format_dataset_name(title))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')

def series_for_metric(df_all, cfg):
    """
    returns a tidy frame with columns:
      train, val, model, value  (value is transformed if needed)
    """
    if cfg.column not in df_all.columns:
        return pd.DataFrame(columns=['train','val','model','value'])
    s = df_all[cfg.column].copy()
    if cfg.transform is not None:
        s = cfg.transform(s)
    out = s.reset_index().rename(columns={cfg.column:'value'})
    return out  # columns: train, val, model, value

def mean_by_val_across_trains(df_metric):
    """
    Returns:
      - per_val_mean: dict[val_name] -> DataFrame(index=model, value=mean across trains)
      - overall_mean: DataFrame(index=model, value=mean across (val, train))
    """
    per_val_mean = {}
    overall_chunks = []
    for val_name, g in df_metric.groupby('val'):
        m = g.groupby('model')['value'].mean().to_frame('value')
        per_val_mean[val_name] = m
        overall_chunks.append(m.rename(columns={'value': val_name}))

    if overall_chunks:
        overall = pd.concat(overall_chunks, axis=1).mean(axis=1).to_frame('value')
    else:
        overall = pd.DataFrame(columns=['value'])

    return per_val_mean, overall

def plot_overall_grid(df_all, metrics):
    """One figure: rows = metrics, columns = each val + 'Overall Mean'."""
    if df_all.empty:
        print("No results found.")
        return

    all_vals = order_vals(df_all.index.get_level_values('val').unique())
    ncols = len(all_vals) + 1
    fig, axs = plt.subplots(len(metrics), ncols, figsize=(4*ncols, 4*len(metrics)))

    for r, (mname, cfg) in enumerate(metrics.items()):
        dfm = series_for_metric(df_all, cfg)
        per_val, overall = mean_by_val_across_trains(dfm)

        for c, val in enumerate(all_vals):
            ax = axs[r, c] if len(metrics) > 1 else axs[c]
            if val not in per_val or per_val[val].empty:
                ax.text(0.5, 0.5, f'{cfg.column}\nnot available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(format_dataset_name(val))
                continue

            m = per_val[val]
            names = order_models(m.index)
            vals = m['value'].to_numpy()
            bars = ax.bar(names, vals, color='dimgrey')
            color_best_worst(bars, vals, cfg.higher_better)
            setup_axis(ax, names, vals, cfg, val)

        axm = axs[r, -1] if len(metrics) > 1 else axs[-1]
        if not overall.empty:
            names = order_models(overall.index)
            vals = overall['value'].to_numpy()
            bars = axm.bar(names, vals, color='black')
            color_best_worst(bars, vals, cfg.higher_better)
            setup_axis(axm, names, vals, cfg, 'Overall Mean')
        else:
            axm.text(0.5, 0.5, f'{cfg.column}\nnot available', ha='center', va='center', transform=axm.transAxes)
            axm.set_title('Overall Mean')

    if len(all_vals) > 0:
        line_x = len(all_vals) / (len(all_vals) + 1)
        fig.add_artist(plt.Line2D([line_x, line_x], [0, 1], color='grey', linewidth=1, linestyle='--',
                                  transform=fig.transFigure, zorder=10))

    legend_elements = [
        Patch(facecolor='limegreen', label='Best'),
        Patch(facecolor='coral', label='Worst'),
        Patch(facecolor='dimgrey', label='Mean across Train Datasets'),
        Patch(facecolor='black', label='Overall Mean'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=4, fontsize=FONT_MEDIUM)
    plt.suptitle("Validation Mean Performance Across All Training Datasets", fontsize=FONT_LARGE, y=1.01)
    
    fig.text(0.5, -0.01, 'Validation Datasets', ha='center', va='bottom', fontsize=FONT_LARGE)
    fig.text(-0.005, 0.5, 'Metrics (← lower is better, → higher is better)', ha='center', va='center', 
             rotation=90, fontsize=FONT_LARGE)
    
    plt.tight_layout()
    plt.show()

def plot_per_train_grids(df_all, metrics):
    """One figure per train set: rows = metrics, columns = each val + 'Mean'."""
    if df_all.empty:
        return

    for train, df_t in df_all.groupby(level='train'):
        vals = order_vals(df_t.index.get_level_values('val').unique())
        ncols = len(vals) + 1
        fig, axs = plt.subplots(len(metrics), ncols, figsize=(4*ncols, 4*len(metrics)))

        per_metric_means = {}
        for mname, cfg in metrics.items():
            dfm = series_for_metric(df_t, cfg)
            per_val, _ = mean_by_val_across_trains(dfm)
            if per_val:
                aligned = pd.concat([v.rename(columns={'value': k}) for k, v in per_val.items()], axis=1)
                per_metric_means[mname] = aligned.mean(axis=1).to_frame('value')
            else:
                per_metric_means[mname] = pd.DataFrame(columns=['value'])

        for r, (mname, cfg) in enumerate(metrics.items()):
            dfm = series_for_metric(df_t, cfg)
            per_val, _ = mean_by_val_across_trains(dfm)

            for c, val in enumerate(vals):
                ax = axs[r, c] if len(metrics) > 1 else axs[c]
                if val not in per_val or per_val[val].empty:
                    ax.text(0.5, 0.5, f'{cfg.column}\nnot available', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(format_dataset_name(val))
                    continue
                m = per_val[val]
                names = order_models(m.index)
                vals_ = m['value'].to_numpy()
                bars = ax.bar(names, vals_, color='darkgrey')
                color_best_worst(bars, vals_, cfg.higher_better)
                setup_axis(ax, names, vals_, cfg, val)

            # mean column
            axm = axs[r, -1] if len(metrics) > 1 else axs[-1]
            mean_df = per_metric_means[mname]
            if not mean_df.empty:
                names = order_models(mean_df.index)
                vals_ = mean_df['value'].to_numpy()
                bars = axm.bar(names, vals_, color='dimgrey')
                color_best_worst(bars, vals_, cfg.higher_better)
                setup_axis(axm, names, vals_, cfg, 'Mean')
            else:
                axm.text(0.5, 0.5, f'{cfg.column}\nnot available', ha='center', va='center', transform=axm.transAxes)
                axm.set_title('Mean')

        if len(vals) > 0:
            line_x = len(vals) / (len(vals) + 1)
            fig.add_artist(plt.Line2D([line_x, line_x], [0, 1], color='darkgrey', linewidth=1, linestyle='--',
                                      transform=fig.transFigure, zorder=10))

        legend_elements = [
            Patch(facecolor='lightgreen', label='Best'),
            Patch(facecolor='lightcoral', label='Worst'),
            Patch(facecolor='darkgrey', label='Others'),
            Patch(facecolor='dimgrey', label='Mean'),
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=4, fontsize=FONT_SMALL)
        plt.suptitle(f"Metrics for YOLO11n-based Models trained on {format_dataset_name(train)} Dataset", fontsize=FONT_LARGE, y=1.01)
        
        # Add common x and y labels
        fig.text(0.5, -0.01, 'Validation Datasets', ha='center', va='bottom', fontsize=FONT_LARGE)
        fig.text(-0.005, 0.5, 'Metrics (← lower is better, → higher is better)', ha='center', va='center', 
                 rotation=90, fontsize=FONT_LARGE)
        
        plt.tight_layout()
        plt.show()

def load_results_data(newest_results='results/detect/data_splits_and_models'):
    here = Path(__file__).resolve().parent
    results_root = here.parent / newest_results
    return load_all_results(results_root)

def create_extended_barplots(df_all, plot_per_train=False):
    # plots average of validation sets over all training sets
    plot_overall_grid(df_all, METRICS)
    if plot_per_train: 
        # plots all validations per training dataset
        plot_per_train_grids(df_all, METRICS)
