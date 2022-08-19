import itertools

from IPython import embed
import os
import typing

# -------------------- Sisyphus --------------------
from sisyphus import gs, tk, Path

# -------------------- Recipes --------------------
import i6_core.rasr as rasr

from i6_experiments.common.setups.rasr import GmmSystem
from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem
import i6_experiments.common.setups.rasr.util as rasr_util

from .config import DIPHONE_CART, DIPHONE_CART_NUM_LABELS
from .config_baseline_hybrid import get_hybrid_system, get_nn_args


def run_hybrid(
    returnn_root: tk.Path, gmm_4gram: GmmSystem, gmm_lstm: GmmSystem
) -> typing.Dict[str, HybridSystem]:
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][9:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    # ******************** HY Init ********************

    corpus_name = "train-other-960"
    lm = {"4gram": gmm_4gram}  # , "lstm": gmm_lstm}
    lr = ["v1", "v2"]
    num_heads = [12, 32]
    sizes = [512, 768]
    n_phone = 2

    results = {}

    for (lm, gmm_sys), conf_size, conf_num_heads, lr in itertools.product(
        lm.items(), sizes, num_heads, lr
    ):
        if conf_size % conf_num_heads != 0:
            print(f"{conf_size} does not work w/ {conf_num_heads} att heads, skipping")
            continue

        name = f"conf-ph:{n_phone}-dim:{conf_size}-h:{conf_num_heads}-lr:{lr}"
        with tk.block(name):
            print(f"hy {name}")
            system = get_hybrid_system(
                n_phones=n_phone,
                gmm_system=gmm_sys,
                corpus_name=corpus_name,
                returnn_root=returnn_root,
                diphone_state_tying_file=DIPHONE_CART,
            )
            nn_args = get_nn_args(
                gmm_system=gmm_sys,
                name=name,
                corpus_name=corpus_name,
                conf_size=conf_size,
                conf_num_heads=conf_num_heads,
                n_phones=n_phone,
                lr=lr,
                diphone_num_out=DIPHONE_CART_NUM_LABELS,
            )

            steps = rasr_util.RasrSteps()
            steps.add_step("nn", nn_args)
            system.run(steps)

            results[name] = system

    return results
