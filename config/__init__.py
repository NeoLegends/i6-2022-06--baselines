from sisyphus import gs, tk, Path

from .config_baseline_fh import run_fh
from .config_baseline_gmm import run_gmm
from .config_baseline_hybrid import run_hybrid


def main():
    with tk.block("gmm"):
        gmm_4gram, gmm_lstm = run_gmm()

    with tk.block("hybrid"):
        run_hybrid(gmm_lstm=gmm_lstm, gmm_4gram=gmm_4gram)

    #with tk.block("fh"):
    #    run_fh()
