from sisyphus import gs, tk, Path

from .config_baseline_fh import run_fh
from .config_baseline_gmm import run_gmm
from .config_baseline_hybrid import run_hybrid

def main():
    run_gmm()
    run_hybrid()
    run_fh()
