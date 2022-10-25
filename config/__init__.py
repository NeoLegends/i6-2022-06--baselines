from sisyphus import gs, tk, Path

import i6_core.tools as tools


def _clone_returnn() -> tk.Path:
    clone_r_job = tools.CloneGitRepositoryJob(
        url="https://github.com/rwth-i6/returnn.git",
        commit="aadac2637ed6ec00925b9debf0dbd3c0ee20d6a6",
        checkout_folder_name="returnn",
    )
    return clone_r_job.out_repository


def gmm_and_hybrid():
    from i6_private.users.gunz.experiments.config_2022_07_baselines import (
        config_10_baseline_gmm_mono as mono,
        config_11_baseline_gmm_di as di,
        config_12_baseline_gmm_tri as tri,
        config_20_baseline_hybrid as hybrid,
        # config_22_baseline_blstm as blyadstm,
    )

    returnn_root = _clone_returnn()

    with tk.block("gmm"):
        mono_sys = mono.run()
        di_sys, cart_di, num_labels_di = di.run()
        tri_sys, cart_tri, num_labels_tri = tri.run()

    with tk.block("hybrid"):
        hybrid.run(
            gmm_mono=mono_sys,
            gmm_di=di_sys,
            gmm_tri=tri_sys,
            returnn_root=returnn_root,
        )
        # blyadstm.run(gmm_4gram=tri_sys, returnn_root=returnn_root)


def fh():
    from i6_private.users.gunz.experiments.config_2022_07_baselines import (
        config_10_baseline_gmm_mono as mono,
        config_30_baseline_fh_mono as fh_mono,
        config_31_baseline_fh_di as fh_di,
        config_32_baseline_fh_tri as fh_tri,
        config_33_baseline_fh_mono_from_mono as fh_mono_from_mono,
    )

    returnn_root = _clone_returnn()

    with tk.block("gmm"):
        mono_sys = mono.run()

    with tk.block("fh"):
        fh_mono.run(returnn_root=returnn_root)
        fh_di.run(returnn_root=returnn_root)
        fh_tri.run(returnn_root=returnn_root)
        fh_mono_from_mono.run(gmm=mono_sys, returnn_root=returnn_root)


def main():
    gmm_and_hybrid()
    fh()
