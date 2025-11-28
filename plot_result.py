
import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Combine metrics into a single 2x2 figure and losses into a single 2x3 figure.
Reads from: runs/train/<exp_name>/results.csv
Outputs:
  - metrics_curve.png
  - loss_curve.png
"""

# ========= Config =========
NAMES = ['YOLOv11s', 'RASF-YOLO']           # experiment folder names under runs/train/
CSV_NAME = 'results.csv'                    # file name in each folder
ROOT = '.'                                  # project root (change if needed)

# Which columns to draw for metrics and losses (title, column-name, ylabel on plot)
METRICS = [
    ('precision',        'metrics/precision(B)',  'precision'),
    ('recall',           'metrics/recall(B)',     'recall'),
    ('mAP_0.5',          'metrics/mAP50(B)',      'mAP_0.5'),
    ('mAP_0.5:0.95',     'metrics/mAP50-95(B)',   'mAP_0.5:0.95'),
]

LOSSES = [
    ('train/box_loss',   'train/box_loss',  'train/box_loss'),
    ('train/cls_loss',   'train/cls_loss',  'train/cls_loss'),
    ('train/dfl_loss',   'train/dfl_loss',  'train/dfl_loss'),
    ('val/box_loss',     'val/box_loss',    'val/box_loss'),
    ('val/cls_loss',     'val/cls_loss',    'val/cls_loss'),
    ('val/dfl_loss',     'val/dfl_loss',    'val/dfl_loss'),
]


# ========= Helpers =========
def read_results(exp_name: str) -> pd.DataFrame:
    """Read one experiment CSV; return empty DataFrame if missing."""
    path = os.path.join(ROOT, 'runs', 'train', exp_name, CSV_NAME)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def get_series(df: pd.DataFrame, col: str) -> np.ndarray:
    """Safe column extraction -> float32 -> replace inf -> interpolate -> numpy array.
       Returns empty array if col missing or all NaN.
    """
    if col not in df.columns:
        return np.array([])
    s = df[col].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    if s.isna().all():
        return np.array([])
    s = s.fillna(s.interpolate()).fillna(method='ffill').fillna(method='bfill')
    return s.to_numpy()


def plot_group(fig, axes, items, title: str):
    """Plot a group (metrics or losses) into provided axes grid.
       items: list of (subplot_title, csv_col, y_label)
    """
    # load dfs once
    dfs = {name: read_results(name) for name in NAMES}

    for ax, (sub_title, col, ylab) in zip(axes, items):
        plotted = False
        for name in NAMES:
            y = get_series(dfs[name], col)
            if y.size > 0:
                x = np.arange(1, len(y)+1)
                ax.plot(x, y, label=name)
                plotted = True
        ax.set_title(sub_title)
        ax.set_xlabel('epoch')
        ax.set_ylabel(ylab)
        ax.set_xlim(left=1)
        if plotted:
            # match the sample figure: legend on lower right for metrics,
            # for losses it's also fine to keep lower right to be consistent
            ax.legend(loc='lower right', frameon=False)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12, alpha=0.7)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(title, y=0.99, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])


# ========= Main =========
def main():
    # Metrics: 2x2
    fig_m, axs_m = plt.subplots(2, 2, figsize=(10, 10), dpi=300)
    plot_group(fig_m, axs_m.ravel(), METRICS, title='Metrics')
    out_m = os.path.join(os.getcwd(), 'metrics_curve.png')
    fig_m.savefig(out_m)
    plt.close(fig_m)

    # Losses: 2x3
    fig_l, axs_l = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
    plot_group(fig_l, axs_l.ravel(), LOSSES, title='Losses')
    out_l = os.path.join(os.getcwd(), 'loss_curve.png')
    fig_l.savefig(out_l)
    plt.close(fig_l)

    print(f'Saved: {out_m}\\nSaved: {out_l}')


if __name__ == '__main__':
    main()