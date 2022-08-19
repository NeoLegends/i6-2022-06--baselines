from sisyphus import gs, tk, Path

import i6_core.tools as tools

from .config_10_baseline_gmm import run_gmm
from .config_20_baseline_di_cart import generate_diphone_cart
from .config_30_baseline_hybrid import run_hybrid
from .config_40_baseline_fh import run_fh


def main():
    # ******************** git setup ********************
    clone_r_job = tools.CloneGitRepositoryJob(
        url="https://github.com/rwth-i6/returnn.git",
        commit="aadac2637ed6ec00925b9debf0dbd3c0ee20d6a6",
        checkout_folder_name="returnn",
    )

    with tk.block("gmm"):
        gmm_4gram, gmm_lstm = run_gmm(
            returnn_root=clone_r_job.out_repository,
            returnn_python_exe=gs.RETURNN_PYTHON_EXE,
        )

    with tk.block("cart-di"):
        diphone_cart, diphone_cart_num_labels = generate_diphone_cart()

    with tk.block("hybrid"):
        run_hybrid(
            diphone_cart=diphone_cart,
            diphone_cart_num_labels=diphone_cart_num_labels,
            gmm_lstm=gmm_lstm,
            gmm_4gram=gmm_4gram,
            returnn_root=clone_r_job.out_repository,
        )

        # run_hybrid_qliang(
        #     returnn_root=clone_r_job.out_repository,
        #     gmm_lstm=gmm_lstm,
        #     gmm_4gram=gmm_4gram,
        # )

    # with tk.block("fh"):
    #    run_fh()
