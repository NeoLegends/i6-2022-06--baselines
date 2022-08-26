from sisyphus import gs, tk, Path

import i6_core.tools as tools

from . import (
    config_10_baseline_gmm as gmm,
    config_11_baseline_di_cart as di_cart,
    config_20_baseline_hybrid as hybrid,
    config_22_baseline_blstm as blyadstm,
    config_30_baseline_fh as fh,
)


def main():
    # ******************** git setup ********************
    clone_r_job = tools.CloneGitRepositoryJob(
        url="https://github.com/rwth-i6/returnn.git",
        commit="aadac2637ed6ec00925b9debf0dbd3c0ee20d6a6",
        checkout_folder_name="returnn",
    )

    with tk.block("gmm"):
        gmm_4gram, gmm_lstm = gmm.run(
            returnn_root=clone_r_job.out_repository,
            returnn_python_exe=gs.RETURNN_PYTHON_EXE,
        )

    with tk.block("cart-di"):
        diphone_cart, diphone_cart_num_labels = di_cart.run()

    with tk.block("hybrid"):
        hybrid.run(
            diphone_cart=diphone_cart,
            diphone_cart_num_labels=diphone_cart_num_labels,
            gmm_lstm=gmm_lstm,
            gmm_4gram=gmm_4gram,
            returnn_root=clone_r_job.out_repository,
        )
        blyadstm.run(gmm_4gram=gmm_4gram, returnn_root=clone_r_job.out_repository)

    # with tk.block("fh"):
    #    fh.run()
