import os

# -------------------- Sisyphus --------------------

from sisyphus import gs, tk, Path

# -------------------- Recipes --------------------

import i6_core.rasr as rasr

import i6_experiments.common.setups.rasr.gmm_system as gmm_system
import i6_experiments.common.setups.rasr.util as rasr_util

import i6_private.users.gunz.setups.ls.pipeline_gmm_args as gmm_setups
import i6_private.users.gunz.setups.ls.pipeline_rasr_args as data_setups


def _run_gmm(
    lm: str, train_data_inputs, dev_data_inputs, test_data_inputs
) -> gmm_system.GmmSystem:
    print(f"GMM {lm} LM")

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
    cart_di_args = gmm_setups.get_cart_args(name="di", phones=2)
    cart_tri_args = gmm_setups.get_cart_args()
    tri_args = gmm_setups.get_triphone_args()
    final_output_args = gmm_setups.get_final_output()

    steps = rasr_util.RasrSteps()
    steps.add_step("extract", init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    steps.add_step("cart_di", cart_di_args)
    steps.add_step("cart_tri", cart_tri_args)
    steps.add_step("tri", tri_args)
    steps.add_step("output", final_output_args)

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

    return lbs_gmm_system


def run_gmm(
    returnn_root: tk.Path, returnn_python_exe: tk.Path
) -> [gmm_system.GmmSystem, gmm_system.GmmSystem]:
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    # ******************** GMM Init ********************

    with tk.block("4gram"):
        train_4gram, dev_4gram, test_4gram = data_setups.get_data_inputs()
        gmm_4gram = _run_gmm("4gram", train_4gram, dev_4gram, test_4gram)

    # with tk.block("lstm"):
    #     train_lstm, dev_lstm, test_lstm = data_setups.get_data_inputs_lstm_lm(
    #         returnn_root=returnn_root, returnn_python_exe=returnn_python_exe
    #     )
    #     gmm_lstm = _run_gmm("lstm", train_lstm, dev_lstm, test_lstm)

    return gmm_4gram, None
