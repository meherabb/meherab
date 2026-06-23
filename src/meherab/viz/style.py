"""ICLR/WACV-standard matplotlib style, color palette, and domain colors used
to produce every figure in the paper.

Key formatting rules encoded in ``ICLR_RC`` (see also ``docs/REPRODUCING.md``
for the full figure-correction protocol): minimum 6.5pt fonts for readable
text, ``pdf.fonttype=42`` vector embedding (so text stays selectable/editable
in the camera-ready PDF, not rasterized), ``savefig.bbox='tight'`` on every
save, and no top/right spines.

Extracted verbatim from the original pipeline, Cell 4.
"""
import matplotlib

ICLR_DW = 5.5  # standard double-column figure width, inches
ICLR_SW = 2.65  # standard sub-panel width, inches
ICLR_H = 2.1  # standard figure height, inches

ICLR_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 8,
    "axes.titleweight": "normal",
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.55,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "grid.linewidth": 0.30,
    "grid.alpha": 0.35,
    "lines.linewidth": 1.1,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Method color palette, used consistently across every figure.
PAL = {
    "lp": "#5C6BC0",  # Linear Probe
    "rr": "#90A4AE",  # Random RASS
    "lora": "#AB47BC",  # LoRA
    "ada": "#26A69A",  # Bottleneck Adapter
    "clip": "#F39C12",  # CLIP-Adapter
    "naswot": "#FFA726",
    "syn": "#EF5350",
    "mhb": "#C0392B",  # MEHERAB
}

# Per-dataset domain colors, used in Fig. 4 / Appendix F.4 / F.6 / F.7.
DOMAIN_COL = {
    "Food-101": "#5C6BC0",
    "Oxford-Pets": "#5C6BC0",
    "Caltech-101": "#5C6BC0",
    "Flowers102": "#5C6BC0",
    "DTD": "#AB47BC",
    "Aircraft": "#26A69A",
    "EuroSAT": "#C0392B",
    "RESISC45": "#C0392B",
    "PatternNet": "#C0392B",
    "UCMerced": "#E67E22",
}


def apply_iclr_style() -> None:
    """Apply the ICLR/WACV rcParams globally. Call once before plotting."""
    matplotlib.rcParams.update(ICLR_RC)
