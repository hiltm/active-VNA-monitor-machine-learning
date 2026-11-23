"""
plot_style.py - IEEE-styled plotting module with Sixers color palette.

style spec:
  - only x and y axes shown (no box / no top+right spines)
  - both axis lines and tick labels are bold
  - axis labels are bold with units in parentheses
  - no title on figures (subfigures use (a), (b), etc. via caption)
  - legend has a bold frame, includes all entries
  - crisp lines with Sixers coloring
  - high DPI output for publication (300+)
  - export in .png, .pdf, .svg

usage:
    import plot_style
    plot_style.apply()  # call once to set global rcParams

    fig, ax = plt.subplots()
    ax.plot(x, y, color=plot_style.colors[0], label='Signal')
    plot_style.format_ax(ax, xlabel='Frequency (GHz)', ylabel='Magnitude (dB)')
    plot_style.save(fig, 'my_figure')

tweaking:
    - edit STYLE dict below to change font sizes, line widths, etc.
    - re-run your plotting script — no MATLAB needed.
    - output SVGs are fully editable in Inkscape/Illustrator.
    - pickle files (.pkl) are saved alongside for re-generating with different params.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pickle

# =============================================================================
# SIXERS COLOR PALETTE (in listed order)
# =============================================================================
SIXERS_COLORS = [
    '#006bb6',   #  1 blue
    '#ed174c',   #  2 red
    '#002b5c',   #  3 navy
    '#007a33',   #  4 green
    '#00a9e0',   #  5 light blue
    '#963821',   #  6 brown
    '#bb9754',   #  7 gold
    '#2d2d2d',   #  8 charcoal
    '#f5b400',   #  9 orange-yellow
    '#324b1e',   # 10 dark olive
    '#fabece',   # 11 light pink
    '#781478',   # 12 dark plum
    '#fff5b4',   # 13 Light yellow
    '#009696',   # 14 medium teal
    '#ffa07a',   # 15 light coral
]

# convenience alias
colors = SIXERS_COLORS

# =============================================================================
# STYLE CONFIGURATION (edit these to tweak all figures at once)
# =============================================================================
STYLE = {
    # Font
    'font_family': 'serif',
    'font_size': 10,           # base font size (pts)
    'label_size': 11,          # axis label font size
    'tick_size': 9,            # tick label font size
    'legend_size': 9,          # legend font size

    # Line widths
    'line_width': 1.5,         # plot line width
    'axis_width': 1.5,         # axis spine line width
    'tick_width': 1.2,         # tick mark width
    'legend_frame_width': 1.2, # legend border width

    # Ticks
    'tick_length': 4,          # tick mark length (pts)
    'tick_direction': 'in',    # 'in', 'out', or 'inout'

    # Figure
    'fig_width': 3.5,          # single-column IEEE width (inches)
    'fig_height': 2.8,         # default height
    'dpi': 300,                # output DPI for raster formats

    # Grid (off by default for IEEE)
    'grid': False,
    'grid_alpha': 0.3,
    'grid_style': '--',
}


# =============================================================================
# APPLY STYLE GLOBALLY
# =============================================================================
def apply():
    """Apply IEEE style to all subsequent matplotlib figures."""
    plt.rcParams.update({
        'font.family': STYLE['font_family'],
        'font.size': STYLE['font_size'],
        'axes.labelsize': STYLE['label_size'],
        'axes.labelweight': 'bold',
        'axes.titlesize': STYLE['label_size'],
        'axes.titleweight': 'bold',
        'axes.linewidth': STYLE['axis_width'],
        'axes.spines.top': False,
        'axes.spines.right': False,

        'xtick.labelsize': STYLE['tick_size'],
        'ytick.labelsize': STYLE['tick_size'],
        'xtick.major.width': STYLE['tick_width'],
        'ytick.major.width': STYLE['tick_width'],
        'xtick.major.size': STYLE['tick_length'],
        'ytick.major.size': STYLE['tick_length'],
        'xtick.direction': STYLE['tick_direction'],
        'ytick.direction': STYLE['tick_direction'],

        'lines.linewidth': STYLE['line_width'],

        'legend.fontsize': STYLE['legend_size'],
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'legend.framealpha': 1.0,
        'legend.fancybox': False,

        'figure.figsize': (STYLE['fig_width'], STYLE['fig_height']),
        'figure.dpi': STYLE['dpi'],
        'savefig.dpi': STYLE['dpi'],
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        'grid.alpha': STYLE['grid_alpha'],
        'grid.linestyle': STYLE['grid_style'],
    })


# =============================================================================
# FORMAT AXES
# =============================================================================
def format_ax(ax, xlabel=None, ylabel=None, legend=True, legend_loc='best',
              grid=None, subfig_label=None):
    """
    Apply IEEE formatting to a single axes object.

    Parameters:
        ax: matplotlib Axes
        xlabel: str, e.g. 'Frequency (GHz)'
        ylabel: str, e.g. 'Magnitude (dB)'
        legend: bool, whether to show legend
        legend_loc: str, legend location
        grid: bool or None (uses STYLE default if None)
        subfig_label: str, e.g. '(a)' — placed in upper-left
    """
    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # bold axis lines
    ax.spines['left'].set_linewidth(STYLE['axis_width'])
    ax.spines['bottom'].set_linewidth(STYLE['axis_width'])

    # bold tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # axis labels
    if xlabel:
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=STYLE['label_size'])
    if ylabel:
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=STYLE['label_size'])

    # no title
    ax.set_title('')

    # legend
    if legend and ax.get_legend_handles_labels()[1]:
        leg = ax.legend(loc=legend_loc, fontsize=STYLE['legend_size'])
        leg.get_frame().set_linewidth(STYLE['legend_frame_width'])
        leg.get_frame().set_edgecolor('black')
        for text in leg.get_texts():
            text.set_fontweight('bold')

    # grid
    show_grid = grid if grid is not None else STYLE['grid']
    if show_grid:
        ax.grid(True, alpha=STYLE['grid_alpha'], linestyle=STYLE['grid_style'])
    else:
        ax.grid(False)

    # subfigure label
    if subfig_label:
        ax.text(0.02, 0.95, subfig_label, transform=ax.transAxes,
                fontsize=STYLE['label_size'], fontweight='bold',
                verticalalignment='top')


# =============================================================================
# SAVE FIGURE
# =============================================================================
def save(fig, basename, output_dir='models', save_pickle=True):
    """
    save figure in .png, .pdf, .svg formats.

    also saves a .pkl file with the figure object for later re-opening/tweaking.
    """
    os.makedirs(output_dir, exist_ok=True)

    for ext in ['png', 'pdf', 'svg']:
        path = os.path.join(output_dir, f'{basename}.{ext}')
        fig.savefig(path, dpi=STYLE['dpi'], bbox_inches='tight',
                    pad_inches=0.05)

    if save_pickle:
        pkl_path = os.path.join(output_dir, f'{basename}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(fig, f)

    print(f"  Saved: {output_dir}/{basename}.{{png,pdf,svg}}" +
          (f" + .pkl" if save_pickle else ""))


# =============================================================================
# UTILITY: Create a standard single-column figure
# =============================================================================
def figure(width=None, height=None, nrows=1, ncols=1, **kwargs):
    """Create a figure with IEEE dimensions."""
    w = width or STYLE['fig_width']
    h = height or STYLE['fig_height']
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), **kwargs)
    return fig, axes


# =============================================================================
# UTILITY: Double-column figure
# =============================================================================
def figure_wide(height=None, nrows=1, ncols=1, **kwargs):
    """Create a double-column IEEE figure (7.16 inches wide)."""
    w = 7.16
    h = height or STYLE['fig_height']
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), **kwargs)
    return fig, axes


# apply on import so any script that does `import plot_style` gets the style
apply()
