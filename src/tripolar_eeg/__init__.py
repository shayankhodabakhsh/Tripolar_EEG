"""tripolar_eeg — analysis pipeline for Tripolar Concentric Ring Electrode (TCRE) EEG.

The package provides a unified loader, preprocessing pipeline, per-subject
analysis, and three-way (Felt vs Gel vs Paste TCRE) comparison framework
backing the manuscript in `paper/`.

Top-level modules:

    eeg_analysis         core engine: ElectrodeConfig, discover/load/analyze,
                         per-channel filters, plotting helpers.
    comparison_analysis  three-way group-level comparison: load_all_subjects
                         with QC gates, extract_type_metrics, statistical tests,
                         plot_three_way_comparison.

Typical usage::

    from tripolar_eeg.eeg_analysis import load_subject, analyze_subject
    from tripolar_eeg.comparison_analysis import load_all_subjects, plot_three_way_comparison

    felt, gel = load_all_subjects("data", "Gel TCRE", felt_recursive=True)
    plot_three_way_comparison(felt, gel, save_dir="output/three_way_comparison")
"""

__version__ = "0.1.0"

from .eeg_analysis import (        # noqa: F401
    FS,
    BANDS,
    BAND_COLORS,
    FELT_TCRE_CONFIG,
    GEL_TCRE_CONFIG,
    discover_subjects,
    load_subject,
    analyze_subject,
    notch_filter,
    fir_bandpass,
    apply_pipeline_variant,
    wavelet_denoise,
)
from .comparison_analysis import (        # noqa: F401
    load_all_subjects,
    extract_type_metrics,
    compare_electrode_types,
    plot_three_way_comparison,
    print_qc_report,
    assign_display_ids,
    TYPE_DISPLAY,
    TYPE_COLORS,
)
