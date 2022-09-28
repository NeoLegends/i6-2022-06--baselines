from sisyphus import gs, tk, Path

import i6_core.tools as tools

from . import (
    config_10_baseline_gmm_mono as mono,
    config_11_baseline_gmm_di as di,
    config_12_baseline_gmm_tri as tri,
    config_20_baseline_hybrid as hybrid,
    config_22_baseline_blstm as blyadstm,
    config_30_baseline_fh_mono as fh_mono,
    config_31_baseline_fh_di as fh_di,
    config_32_baseline_fh_tri as fh_tri,
)


def main():
    # ******************** git setup ********************
    clone_r_job = tools.CloneGitRepositoryJob(
        url="https://github.com/rwth-i6/returnn.git",
        commit="aadac2637ed6ec00925b9debf0dbd3c0ee20d6a6",
        checkout_folder_name="returnn",
    )

    with tk.block("gmm"):
        mono_sys = mono.run()
        di_sys, cart_di, num_labels_di = di.run()
        tri_sys, cart_tri, num_labels_tri = tri.run()

    with tk.block("hybrid"):
        hybrid.run(
            gmm_mono=mono_sys,
            gmm_di=di_sys,
            gmm_tri=tri_sys,
            returnn_root=clone_r_job.out_repository,
        )
        blyadstm.run(gmm_4gram=tri_sys, returnn_root=clone_r_job.out_repository)

    with tk.block("fh"):
        fh_mono.run(gmm=mono_sys, returnn_root=clone_r_job.out_repository)
        fh_di.run(returnn_root=clone_r_job.out_repository)
        fh_tri.run(returnn_root=clone_r_job.out_repository)
