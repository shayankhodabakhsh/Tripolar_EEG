"""Read-only EEG quality review package.

The package deliberately keeps technical QC, human decisions, and post-review
physiology in separate modules.  Importing it never reads or writes raw data.
"""

__version__ = "0.1.0"
