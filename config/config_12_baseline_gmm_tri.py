import os

# -------------------- Sisyphus --------------------
import typing

from i6_core.meta import CartAndLDA
from sisyphus import gs, tk, Path

# -------------------- Recipes --------------------

import i6_core.rasr as rasr

import i6_experiments.common.setups.rasr.gmm_system as gmm_system
import i6_experiments.common.setups.rasr.util as rasr_util

import i6_private.users.gunz.setups.ls.pipeline_gmm_args as gmm_setups
import i6_private.users.gunz.setups.ls.pipeline_rasr_args as data_setups


def _run_gmm(
    train_data_inputs, dev_data_inputs, test_data_inputs
) -> typing.Tuple[gmm_system.GmmSystem, tk.Path, int]:
    print(f"GMM Tri")

    mfcc_cepstrum_options = {
        "normalize": False,
        "outputs": 16,
        "add_epsilon": True,
        "epsilon": 1e-10,
    }

    gt_options_extra_args = {
        "normalize": False,
    }

    init_args = data_setups.get_init_args(
        dc_detection=False,
        mfcc_cepstrum_options=mfcc_cepstrum_options,
        gt_options_extra_args=gt_options_extra_args,
    )

    mono_args = gmm_setups.get_monophone_args(allow_zero_weights=True)
    cart_tri_args = gmm_setups.get_cart_args()
    tri_args = gmm_setups.get_triphone_args()
    tri_output_args = gmm_setups.get_final_output()

    steps = rasr_util.RasrSteps()
    steps.add_step("extract", init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    steps.add_step("cart", cart_tri_args)
    steps.add_step("tri", tri_args)
    steps.add_step("output", tri_output_args)

    # ******************** GMM System ********************

    rasr_path = os.path.join(gs.RASR_ROOT, "arch", gs.RASR_ARCH)
    lbs_gmm_system = gmm_system.GmmSystem(rasr_binary_path=tk.Path(rasr_path))
    lbs_gmm_system.init_system(
        rasr_init_args=init_args,
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )
    lbs_gmm_system.run(steps)

    cart_lda: CartAndLDA = lbs_gmm_system.jobs["train-other-960"][
        "cart_and_lda_train-other-960_mono"
    ]
    return lbs_gmm_system, cart_lda.last_cart_tree, cart_lda.last_num_cart_labels


def run() -> typing.Tuple[gmm_system.GmmSystem, tk.Path, int]:
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    # ******************** GMM Init ********************

    with tk.block("tri"):
        train_4gram, dev_4gram, test_4gram = data_setups.get_data_inputs()
        return _run_gmm(train_4gram, dev_4gram, test_4gram)
