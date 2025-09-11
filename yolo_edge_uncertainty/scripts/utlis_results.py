from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from dataclasses import dataclass
from typing import Callable, Optional
import os

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

DEFAULT_MODEL_SELECTION = [
    'Baseline', 'Ensemble', 'MC Dropout', 'EDL MEH'
]

MODEL_ORDER_PRETTY = [
    'Base Pretrained', 'Baseline', 'Base Uncertainty',
    'Ensemble', 'MC Dropout', 'EDL MEH'
]

def order_models(idx_like):
    """Return default-selected models in fixed order, dropping any missing."""
    present = set(idx_like)
    return [m for m in DEFAULT_MODEL_SELECTION if m in present]

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
    'base-confidence': 'Baseline',
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
    'mAP': MetricCfg('metrics/mAP50(B)', 'mAP →', (0, 0.55), True),
    'Precision': MetricCfg('metrics/precision(B)', 'Precision →', (0, 1.0), True),
    'Recall': MetricCfg('metrics/recall(B)', 'Recall →', (0, 0.55), True),
    'mUE': MetricCfg('metrics/mUE50', '← mUE', (0, 0.55), False),
    'AUROC': MetricCfg('metrics/AUROC50', 'AUROC →', (0.5, 1.0), True),
    'FPR95': MetricCfg('metrics/FPR95_50', '← FPR95', (0, 1.0), False),
    'E-AURC': MetricCfg('metrics/E-AURC50', '← E-AURC', (0, 0.12), False),
    'FPS': MetricCfg('speed_val_sum', 'FPS (@A100) →', (0, None), True, transform=lambda s: 1000.0 / s),  # ms → FPS
}

def format_dataset_name(name):
    """Pretty-print raw dataset identifier to human-readable name (no abbreviations)."""
    return (name.replace('-from-coco80', '')
                .replace('-coco80', '')
                .replace('-', ' ')
                .title()
                .replace('Raincityscapes', 'RainCityscapes')
                .replace('Nuimages', 'nuImages')
                .replace('Bdd100K', 'BDD100k')
                .replace('Kitti', 'KITTI'))

def _abbr_dataset_pretty_name(pretty: str) -> str:
    """Abbreviate long dataset names for compact table headers only."""
    return (pretty
            .replace('Foggy Cityscapes', 'Foggy C.')
            .replace('RainCityscapes', 'RainC.'))

def _get_base_dir() -> Path:
    """Ensure and return the base directory 'yolo_edge_uncertainty' at repo root.

    Robustly finds the repository root (searching for common markers) and
    creates/returns '<repo_root>/yolo_edge_uncertainty'.
    """
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name == 'yolo_edge_uncertainty':
            p.mkdir(exist_ok=True)
            return p
    def find_repo_root(start: Path) -> Path:
        for p in start.parents:
            if (p / 'pyproject.toml').exists() or (p / '.git').exists() or (p / 'README.md').exists():
                return p
        return start.parents[2] if len(start.parents) >= 3 else start.parents[-1]

    repo_root = find_repo_root(here)
    base = repo_root / 'yolo_edge_uncertainty'
    base.mkdir(exist_ok=True)
    return base

def get_figures_dir() -> Path:
    """Return 'yolo_edge_uncertainty/figures', creating it if needed."""
    base = _get_base_dir()
    figures_dir = base / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir

def get_tables_dir() -> Path:
    """Return 'yolo_edge_uncertainty/tables', creating it if needed."""
    base = _get_base_dir()
    tables_dir = base / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir

def get_csv_dir() -> Path:
    """Return 'yolo_edge_uncertainty/csv', creating it if needed."""
    base = _get_base_dir()
    csv_dir = base / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir

def save_figure(fig, filename, dpi=300):
    """Save figure to the figures directory with multiple formats."""
    figures_dir = get_figures_dir()
    base_name = Path(filename).stem
    fig.savefig(figures_dir / f'{base_name}.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(figures_dir / f'{base_name}.pdf', bbox_inches='tight')
    print(f"📁 Saved figure: {base_name}.png and {base_name}.pdf")

def _sum_val_speed(df):
    speed_cols = [c for c in df.columns if 'speed' in c if not 'loss' in c] # inference speed only
    return df[speed_cols].sum(axis=1) if speed_cols else pd.Series([np.nan]*len(df), index=df.index)

def load_all_results(results_root, exclude_val_datasets=None, save_csvs=True,
                     include_models=None, model_rename=None):
    """
    returns a tidy df with MultiIndex (train,val,model) and result columns,
    including computed 'speed_val_sum'. Optionally saves CSV files per train dataset.
    
    Args:
        results_root: Path to results directory
        exclude_val_datasets: List of validation dataset names to exclude (e.g., ['foggy-cityscapes-from-coco80'])
        save_csvs: Whether to save CSV files per train dataset and overall mean
        include_models: Optional iterable of models to include (accepts raw ids like
            'mc-dropout' or pretty names like 'MC Dropout'). If None, include all known.
        model_rename: Optional dict mapping raw model ids to custom display names.
    """
    if exclude_val_datasets is None:
        exclude_val_datasets = []
    
    rename_map = dict(MODEL_RENAME)
    if model_rename:
        rename_map.update(model_rename)
    reverse_rename = {v: k for k, v in rename_map.items()}
    default_include_raw = {'base-confidence', 'ensemble', 'mc-dropout', 'edl-meh'}
    include_raw = set()
    if include_models is None:
        include_raw = set(default_include_raw)
    else:
        for m in include_models:
            if m in MODEL_ORDER_RAW:
                include_raw.add(m)
            elif m in reverse_rename:
                include_raw.add(reverse_rename[m])
            else:
                include_raw.add(m)
    
    rows = []
    train_dfs = {}
    
    for train_dir in sorted([p for p in results_root.iterdir() if p.is_dir() and p.name.startswith('train-')]):
        train = train_dir.name.replace('train-', '')
        train_rows = []
        
        for val_dir in sorted([p for p in train_dir.iterdir() if p.is_dir() and p.name.startswith('val-')]):
            val = val_dir.name.replace('val-', '')
            
            if val in exclude_val_datasets:
                continue
            for model_dir in sorted([p for p in val_dir.iterdir() if p.is_dir()]):
                csv_path = model_dir / 'results_extended.csv'
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path)
                if 'name' not in df.columns:
                    continue
                df = df.set_index('name')
                available = [m for m in MODEL_ORDER_RAW if m in df.index]
                if include_raw is not None:
                    available = [m for m in available if m in include_raw]
                if not available:
                    continue
                df = df.loc[available].copy()
                df['speed_val_sum'] = _sum_val_speed(df)
                df['__train'] = train
                df['__val'] = val
                df['__model'] = df.index
                rows.append(df)
                train_rows.append(df)
        
        if train_rows:
            train_df = pd.concat(train_rows, axis=0, ignore_index=False)
            train_df['model'] = train_df['__model'].map(rename_map).fillna(train_df['__model'])
            idx = pd.MultiIndex.from_arrays([train_df['__train'], train_df['__val'], train_df['model']],
                                            names=['train', 'val', 'model'])
            train_df = train_df.drop(columns=['__train','__val','__model','model'])
            train_df = train_df.set_index(idx)
            train_dfs[train] = train_df

    if not rows:
        return pd.DataFrame()

    df_all = pd.concat(rows, axis=0, ignore_index=False)
    df_all['model'] = df_all['__model'].map(rename_map).fillna(df_all['__model'])
    idx = pd.MultiIndex.from_arrays([df_all['__train'], df_all['__val'], df_all['model']],
                                    names=['train', 'val', 'model'])
    df_all = df_all.drop(columns=['__train','__val','__model','model'])
    df_all = df_all.set_index(idx)
    
    if save_csvs:
        csv_dir = get_csv_dir()
        
        for train_name, train_df in train_dfs.items():
            csv_filename = f'results_train_{train_name.replace("-", "_")}.csv'
            csv_path = csv_dir / csv_filename
            train_df.to_csv(csv_path)
            print(f"💾 Saved CSV: {csv_filename}")
        overall_mean_data = {}
        for metric_name, cfg in METRICS.items():
            dfm = series_for_metric(df_all, cfg)
            if not dfm.empty:
                per_val, overall_mean = mean_by_val_across_trains(dfm)
                if not overall_mean.empty:
                    overall_mean_data[metric_name] = overall_mean['value']
        
        if overall_mean_data:
            overall_df = pd.DataFrame(overall_mean_data)
            overall_csv_path = csv_dir / 'results_overall_mean.csv'
            overall_df.to_csv(overall_csv_path)
            print(f"💾 Saved CSV: results_overall_mean.csv")
    
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
    returns a tidy frame
    """
    if cfg.column not in df_all.columns:
        return pd.DataFrame(columns=['train','val','model','value'])
    s = df_all[cfg.column].copy()
    if cfg.transform is not None:
        s = cfg.transform(s)
    out = s.reset_index().rename(columns={cfg.column:'value'})
    return out

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

def plot_single_bar(ax, data_df, cfg, title, bar_color='dimgrey'):
    """Plot a single bar chart on the given axis."""
    if data_df.empty:
        ax.text(0.5, 0.5, f'{cfg.column}\nnot available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(format_dataset_name(title))
        return
    
    names = order_models(data_df.index)
    if not names:
        names = list(data_df.index)
    
    ordered_data = data_df.loc[names] if len(names) == len(data_df) else data_df
    values = ordered_data['value'].to_numpy()
    
    bars = ax.bar(names, values, color=bar_color)
    color_best_worst(bars, values, cfg.higher_better)
    setup_axis(ax, names, values, cfg, title)

def add_separator_line(fig, n_main_cols, total_cols, line_color='grey'):
    """Add a vertical separator line before the last column."""
    if n_main_cols > 0:
        line_x = n_main_cols / total_cols
        fig.add_artist(plt.Line2D([line_x, line_x], [0, 1], color=line_color, linewidth=1, linestyle='--',
                                  transform=fig.transFigure, zorder=10))

def add_common_labels(fig, title, legend_elements, legend_fontsize=FONT_MEDIUM):
    """Add common figure labels and legend."""
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=4, fontsize=legend_fontsize)
    plt.suptitle(title, fontsize=FONT_LARGE, y=1.01)
    fig.text(0.5, -0.01, 'Validation Datasets', ha='center', va='bottom', fontsize=FONT_LARGE)
    fig.text(-0.005, 0.5, 'Metrics (← lower is better, → higher is better)', ha='center', va='center', 
             rotation=90, fontsize=FONT_LARGE)

def create_metrics_grid(df_data, metrics, vals_list, mean_data_dict, config):
    """Create a metrics grid with common structure for both overall and per-train plots."""
    ncols = len(vals_list) + 1
    fig, axs = plt.subplots(len(metrics), ncols, figsize=(4*ncols, 4*len(metrics)))
    
    for r, (mname, cfg) in enumerate(metrics.items()):
        dfm = series_for_metric(df_data, cfg)
        per_val, overall_or_mean = mean_by_val_across_trains(dfm)
        for c, val in enumerate(vals_list):
            ax = axs[r, c] if len(metrics) > 1 else axs[c]
            val_data = per_val.get(val, pd.DataFrame(columns=['value']))
            plot_single_bar(ax, val_data, cfg, val, config['val_color'])
        axm = axs[r, -1] if len(metrics) > 1 else axs[-1]
        mean_col_data = mean_data_dict.get(mname, overall_or_mean) if mean_data_dict else overall_or_mean
        mean_title = config['mean_title']
        plot_single_bar(axm, mean_col_data, cfg, mean_title, config['mean_color'])
    
    add_separator_line(fig, len(vals_list), ncols, config.get('line_color', 'grey'))
    add_common_labels(fig, config['title'], config['legend_elements'], config.get('legend_fontsize', FONT_MEDIUM))
    plt.tight_layout()
    
    if 'filename' in config:
        save_figure(fig, config['filename'])
    
    plt.show()

def plot_overall_grid(df_all, metrics, title=None, filename=None):
    """One figure: rows = metrics, columns = each val + 'Overall Mean'.

    Args:
        df_all: DataFrame with MultiIndex (train, val, model)
        metrics: dict of metric_name -> MetricCfg to plot
        title: optional custom title
        filename: optional custom base filename (without extension)
    """
    if df_all.empty:
        print("No results found.")
        return

    all_vals = order_vals(df_all.index.get_level_values('val').unique())
    
    config = {
        'val_color': 'dimgrey',
        'mean_color': 'black', 
        'mean_title': 'Overall Mean',
        'title': title or "Validation Mean Performance Across All Training Datasets",
        'line_color': 'grey',
        'legend_elements': [
            Patch(facecolor='limegreen', label='Best'),
            Patch(facecolor='coral', label='Worst'),
            Patch(facecolor='dimgrey', label='Mean across Train Datasets'),
            Patch(facecolor='black', label='Overall Mean'),
        ],
        'legend_fontsize': FONT_MEDIUM,
        'filename': filename or 'overall_validation_performance_grid'
    }
    
    create_metrics_grid(df_all, metrics, all_vals, None, config)

def plot_per_train_grids(df_all, metrics):
    """One figure per train set: rows = metrics, columns = each val + 'Mean'."""
    if df_all.empty:
        return

    for train, df_t in df_all.groupby(level='train'):
        vals = order_vals(df_t.index.get_level_values('val').unique())
        
        per_metric_means = {}
        for mname, cfg in metrics.items():
            dfm = series_for_metric(df_t, cfg)
            per_val, _ = mean_by_val_across_trains(dfm)
            if per_val:
                aligned = pd.concat([v.rename(columns={'value': k}) for k, v in per_val.items()], axis=1)
                per_metric_means[mname] = aligned.mean(axis=1).to_frame('value')
            else:
                per_metric_means[mname] = pd.DataFrame(columns=['value'])
        
        config = {
            'val_color': 'darkgrey',
            'mean_color': 'dimgrey',
            'mean_title': 'Mean',
            'title': f"Metrics for YOLO11n-based Models trained on {format_dataset_name(train)} Dataset",
            'line_color': 'darkgrey',
            'legend_elements': [
                Patch(facecolor='lightgreen', label='Best'),
                Patch(facecolor='lightcoral', label='Worst'),
                Patch(facecolor='darkgrey', label='Others'),
                Patch(facecolor='dimgrey', label='Mean'),
            ],
            'legend_fontsize': FONT_SMALL,
            'filename': f'train_{train.replace("-", "_")}_validation_performance_grid'
        }
        
        create_metrics_grid(df_t, metrics, vals, per_metric_means, config)

def plot_per_train_grids_split(df_all):
    """Two figures per train set, split by metric group.

    Part 1: mAP, Precision, Recall, FPS
    Part 2: mUE, AUROC, FPR95, E-AURC
    """
    if df_all.empty:
        return

    part1_names = ['mAP', 'Precision', 'Recall', 'FPS']
    part2_names = ['mUE', 'AUROC', 'FPR95', 'E-AURC']

    def subset(names):
        return {name: METRICS[name] for name in names if name in METRICS}

    for train, df_t in df_all.groupby(level='train'):
        vals = order_vals(df_t.index.get_level_values('val').unique())

        per_metric_means_all = {}
        for mname, cfg in METRICS.items():
            dfm = series_for_metric(df_t, cfg)
            per_val, _ = mean_by_val_across_trains(dfm)
            if per_val:
                aligned = pd.concat([v.rename(columns={'value': k}) for k, v in per_val.items()], axis=1)
                per_metric_means_all[mname] = aligned.mean(axis=1).to_frame('value')
            else:
                per_metric_means_all[mname] = pd.DataFrame(columns=['value'])

        metrics_part1 = subset(part1_names)
        per_metric_means_part1 = {k: per_metric_means_all.get(k, pd.DataFrame(columns=['value'])) for k in metrics_part1.keys()}
        config1 = {
            'val_color': 'darkgrey',
            'mean_color': 'dimgrey',
            'mean_title': 'Mean',
            'title': f"Metrics for YOLO11n-based Models trained on the {format_dataset_name(train)} Dataset",
            'line_color': 'darkgrey',
            'legend_elements': [
                Patch(facecolor='lightgreen', label='Best'),
                Patch(facecolor='lightcoral', label='Worst'),
                Patch(facecolor='darkgrey', label='Others'),
                Patch(facecolor='dimgrey', label='Mean'),
            ],
            'legend_fontsize': FONT_SMALL,
            'filename': f"train_{train.replace('-', '_')}_validation_performance_grid_part1"
        }
        create_metrics_grid(df_t, metrics_part1, vals, per_metric_means_part1, config1)

        metrics_part2 = subset(part2_names)
        per_metric_means_part2 = {k: per_metric_means_all.get(k, pd.DataFrame(columns=['value'])) for k in metrics_part2.keys()}
        config2 = {
            'val_color': 'darkgrey',
            'mean_color': 'dimgrey',
            'mean_title': 'Mean',
            'title': f"Metrics for YOLO11n-based Models trained on the {format_dataset_name(train)} Dataset)",
            'line_color': 'darkgrey',
            'legend_elements': [
                Patch(facecolor='lightgreen', label='Best'),
                Patch(facecolor='lightcoral', label='Worst'),
                Patch(facecolor='darkgrey', label='Others'),
                Patch(facecolor='dimgrey', label='Mean'),
            ],
            'legend_fontsize': FONT_SMALL,
            'filename': f"train_{train.replace('-', '_')}_validation_performance_grid_part2"
        }
        create_metrics_grid(df_t, metrics_part2, vals, per_metric_means_part2, config2)

def load_results_data(newest_results='results/detect/data_splits_and_models', exclude_val_datasets=None,
                      save_csvs=True, include_models=None, model_rename=None):
    """
    Load results data with optional validation dataset filtering.
    
    Args:
        newest_results: Path to results directory
        exclude_val_datasets: List of validation dataset names to exclude (e.g., ['foggy-cityscapes-from-coco80'])
        save_csvs: Whether to save CSV files per train dataset and overall mean
    """
    here = Path(__file__).resolve().parent.parent
    results_root = here.parent / newest_results
    return load_all_results(
        results_root,
        exclude_val_datasets=exclude_val_datasets,
        save_csvs=save_csvs,
        include_models=include_models,
        model_rename=model_rename,
    )

def create_extended_barplots(df_all, plot_per_train=False):
    """Create barplot grids, splitting overall figure into two parts by metric group.

    Part 1: mAP, Precision, Recall, FPS
    Part 2: mUE, AUROC, FPR95, E-AURC
    """
    part1_names = ['mAP', 'Precision', 'Recall', 'FPS']
    part2_names = ['mUE', 'AUROC', 'FPR95', 'E-AURC']

    def subset(names):
        return {name: METRICS[name] for name in names if name in METRICS}
    
    metrics_part1 = subset(part1_names)
    metrics_part2 = subset(part2_names)
    plot_overall_grid(
        df_all,
        metrics_part1,
        title="Validation Performance (mAP, Precision, Recall, FPS)",
        filename="overall_validation_performance_grid_part1",
    )
    plot_overall_grid(
        df_all,
        metrics_part2,
        title="Validation Performance (Uncertainty Metrics)",
        filename="overall_validation_performance_grid_part2",
    )

    if plot_per_train:
        plot_per_train_grids_split(df_all)

def _get_train_info(df):
    """Get training dataset info string."""
    if 'train' not in df.columns:
        return "trained on single dataset"
    
    unique_trains = df['train'].nunique()
    if unique_trains > 1:
        return "mean of all training runs"
    
    train_name = df['train'].iloc[0]
    formatted_train = format_dataset_name(train_name)
    return f"trained on {formatted_train} dataset"

def _get_metric_mapping(df):
    """Get mapping from metric columns to display names with arrows."""
    mapping = {}
    for metric_name, cfg in METRICS.items():
        if cfg.column in df.columns:
            arrow = "↑" if cfg.higher_better else "↓"
            mapping[cfg.column] = f"{metric_name} [{arrow}]"
    return mapping

def _apply_transforms_and_rename(df, metric_mapping):
    """Apply transforms and rename columns."""
    df = df[list(metric_mapping.keys())].copy()
    
    for col, display_name in metric_mapping.items():
        original_name = display_name.split(' [')[0]
        cfg = METRICS[original_name]
        if cfg.transform:
            df[col] = cfg.transform(df[col])
    
    return df.rename(columns=metric_mapping)

def _format_and_bold(df, precision):
    """Format values and bold the best ones."""
    df_str = df.copy().astype(str)
    for col in df.columns:
        base_name = col.split(' [')[0]
        prec = 1 if base_name == 'FPS' else precision
        for idx in df.index:
            df_str.loc[idx, col] = f"{df.loc[idx, col]:.{prec}f}"
    if isinstance(df.index, pd.MultiIndex):
        for val_dataset in df.index.get_level_values(0).unique():
            val_data = df.loc[val_dataset]
            for col in df.columns:
                original_name = col.split(' [')[0]
                cfg = METRICS[original_name]
                best_idx = val_data[col].idxmax() if cfg.higher_better else val_data[col].idxmin()
                best_val = val_data.loc[best_idx, col]
                prec = 1 if original_name == 'FPS' else precision
                df_str.loc[(val_dataset, best_idx), col] = f"\\textbf{{{best_val:.{prec}f}}}"
    else:
        for col in df.columns:
            original_name = col.split(' [')[0]
            cfg = METRICS[original_name]
            best_idx = df[col].idxmax() if cfg.higher_better else df[col].idxmin()
            best_val = df.loc[best_idx, col]
            prec = 1 if original_name == 'FPS' else precision
            df_str.loc[best_idx, col] = f"\\textbf{{{best_val:.{prec}f}}}"
    
    return df_str

def _build_latex_tabular(df_formatted):
    """Build LaTeX tabular with a single header row: Metric, Model, <datasets> (+ Mean).
    """
    is_multiindex = isinstance(df_formatted.index, pd.MultiIndex)

    def _col_spec(n_cols: int, has_mean: bool) -> str:
        if n_cols <= 0:
            return ''
        cols = ['l'] * n_cols
        if has_mean and n_cols >= 2:
            return ''.join(cols[:-1]) + '|' + cols[-1]
        return ''.join(cols)
    def _has_mean(cols) -> bool:
        return len(cols) > 0 and str(list(cols)[-1]).strip().lower() == 'mean'

    if is_multiindex:
        data_cols = list(df_formatted.columns)
        col_format = _col_spec(2 + len(data_cols), _has_mean(data_cols))
        lines = [f'\\begin{{tabular}}{{{col_format}}}', '\\hline']
        idx_names = df_formatted.index.names or ['Metric', 'Model']
        name0 = idx_names[0] or 'Metric'
        name1 = idx_names[1] or 'Model'
        header_parts = [name0, name1] + data_cols
        header_line = ' & '.join(header_parts) + ' \\\\ \\hline'
        lines.append(header_line)
        current_metric = None
        for (metric_name, model_name), row in df_formatted.iterrows():
            if metric_name != current_metric:
                if current_metric is not None:
                    lines.append('\\midrule')
                current_metric = metric_name
                metric_label = str(metric_name)
            else:
                metric_label = ''

            row_parts = [metric_label, str(model_name)] + [str(val) for val in row.values]
            lines.append(' & '.join(row_parts) + ' \\\\')
    else:
        data_cols = list(df_formatted.columns)
        col_format = _col_spec(1 + len(data_cols), _has_mean(data_cols))
        lines = [f'\\begin{{tabular}}{{{col_format}}}', '\\hline']

        header_parts = [df_formatted.index.name or 'Model'] + data_cols
        header_line = ' & '.join(header_parts) + ' \\\\ \\hline'
        lines.append(header_line)

        for idx, row in df_formatted.iterrows():
            row_parts = [str(idx)] + [str(val) for val in row.values]
            lines.append(' & '.join(row_parts) + ' \\\\')

    lines.append('\\end{tabular}')
    return '\n'.join(lines)

def csv_to_latex_table(csv_path, output_path=None, precision=3, include_models=None, model_rename=None):
    """Convert CSV to LaTeX table.
    """
    df = pd.read_csv(csv_path)
    if 'model' in df.columns:
        if model_rename:
            df['model'] = df['model'].replace(model_rename)
    train_info = _get_train_info(df)
    metric_mapping = _get_metric_mapping(df)
    
    if not metric_mapping:
        raise ValueError("No valid metrics found in CSV")
    
    if 'train' in df.columns and 'val' in df.columns and 'model' in df.columns:
        df = df.set_index(['val', 'model'])
        df = _apply_transforms_and_rename(df, metric_mapping)
        val_mapping_pretty = {v: format_dataset_name(v) for v in df.index.levels[0]}
        val_mapping = {k: _abbr_dataset_pretty_name(v) for k, v in val_mapping_pretty.items()}
        formatted_order = [
            _abbr_dataset_pretty_name(format_dataset_name(v))
            for v in VAL_ORDER_RAW
            if _abbr_dataset_pretty_name(format_dataset_name(v)) in val_mapping.values()
        ]
        ordered_models = order_models(df.index.get_level_values(1).unique())
        if include_models:
            include_set = set(include_models)
            ordered_models = [m for m in ordered_models if m in include_set]
        
        desired_metric_order = ['mAP', 'Precision', 'Recall', 'FPS', 'mUE', 'AUROC', 'FPR95', 'E-AURC']
        base_to_col = {col.split(' [')[0]: col for col in df.columns}
        ordered_metric_cols = [base_to_col[m] for m in desired_metric_order if m in base_to_col]

        df_pivoted_list = []
        for metric_col in ordered_metric_cols:
            metric_name = metric_col.split(' [')[0]
            cfg = METRICS[metric_name]
            arrow = "↑" if cfg.higher_better else "↓"
            metric_display = f"{metric_name} [{arrow}]"
            df_metric = df[metric_col].reset_index()
            df_metric['val'] = df_metric['val'].map(val_mapping)
            df_metric = df_metric.pivot(index='model', columns='val', values=metric_col)
            df_metric = df_metric.reindex(index=ordered_models, columns=formatted_order)
            df_metric.index = pd.MultiIndex.from_product([[metric_display], df_metric.index], names=['Metric', 'Model'])
            df_pivoted_list.append(df_metric)
        
        df = pd.concat(df_pivoted_list)
        df.index.names = ['Metric', 'Model']
        df['Mean'] = df.mean(axis=1, skipna=True)
        
        df_formatted = df.copy().astype(str)
        for metric_label in df.index.get_level_values('Metric').unique():
            metric_data = df.loc[metric_label]
            base_metric_name = str(metric_label).split(' [')[0]
            cfg = METRICS[base_metric_name]
            prec = 1 if base_metric_name == 'FPS' else precision
            
            for col in df.columns:
                if not metric_data[col].isna().all():
                    best_idx = metric_data[col].idxmax() if cfg.higher_better else metric_data[col].idxmin()
                    best_val = metric_data.loc[best_idx, col]
                    if not pd.isna(best_val):
                        df_formatted.loc[(metric_label, best_idx), col] = f"\\textbf{{{best_val:.{prec}f}}}"
            for model in metric_data.index:
                for col in df.columns:
                    val = metric_data.loc[model, col]
                    if not pd.isna(val) and not df_formatted.loc[(metric_label, model), col].startswith('\\textbf'):
                        df_formatted.loc[(metric_label, model), col] = f"{val:.{prec}f}"
    
    elif 'model' in df.columns:
        df = df.set_index('model')
        df.index.name = 'Model'
        ordered_models = order_models(df.index.unique())
        if include_models:
            include_set = set(include_models)
            ordered_models = [m for m in ordered_models if m in include_set]
        df = df.reindex(ordered_models)
        
        df = _apply_transforms_and_rename(df, metric_mapping)
        df_formatted = _format_and_bold(df, precision)
    
    latex_str = _build_latex_tabular(df_formatted)
    
    caption = (
        f"Validation performance results for models {train_info}. "
        "Arrows indicate optimization direction (↑ higher is better, ↓ lower is better). "
        "Best values are shown in \\textbf{bold}. "
        "Columns are validation datasets; the last column (after |) is the Mean across datasets. "
        "Abbrev.: Foggy C.=Foggy Cityscapes, RainC.=RainCityscapes."
    )
    
    if "trained on" in train_info and "dataset" in train_info:
        dataset_part = train_info.split("trained on ")[1].split(" dataset")[0].lower().replace(" ", "_")
        label = f"tab:results_{dataset_part}"
    else:
        label = "tab:results_mean"
    
    table_latex = f"""\\begin{{table}}[ht]
  \\centering
  {latex_str.strip()}
  \\caption{{{caption}}}
  \\label{{{label}}}
\\end{{table}}"""
    
    if not output_path:
        default_name = f"{Path(csv_path).stem}.tex"
        output_path = get_tables_dir() / default_name
    
    with open(output_path, 'w') as f:
        f.write(table_latex)
    try:
        rel = Path(output_path).relative_to(_get_base_dir().parent)
    except Exception:
        rel = Path(output_path).name
    print(f"💾 Saved LaTeX table: {rel}")
    
    return table_latex
