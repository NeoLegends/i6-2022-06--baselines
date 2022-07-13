import os
import typing

# -------------------- Sisyphus --------------------

from sisyphus import gs, tk, Path

# -------------------- Recipes --------------------

import i6_core.rasr as rasr

import i6_experiments.common.setups.rasr.hybrid_system as hybrid_system
import i6_experiments.common.setups.rasr.gmm_system as gmm_system
import i6_experiments.common.setups.rasr.util as rasr_util
import i6_experiments.users.raissi.experiments.librispeech.data_preparation.other_960h.pipeline_base_args as data_setups
import i6_experiments.users.raissi.experiments.librispeech.data_preparation.other_960h.pipeline_gmm_args as gmm_setups


def n_phones_to_str(n_phones: int) -> str:
    if n_phones == 1:
        return "mono"
    elif n_phones == 2:
        return "di"
    elif n_phones == 3:
        return "tri"
    else:
        raise ValueError(f"n_phones must be either 1, 2 or 3, not {n_phones}")


def _run_hybrid(
    lm: str,
    n_phones: int,
    gmm_system: gmm_system.GmmSystem,
) -> hybrid_system.HybridSystem:
    print(f"Hybrid {n_phones_to_str(n_phones)} {lm}")

    lbs_hy_system = hybrid_system.HybridSystem()
    return lbs_hy_system


def run_hybrid(gmm_4gram, gmm_lstm) -> typing.Dict[(int, str), hybrid_system.HybridSystem]:
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    # ******************** GMM Init ********************

    n_phones = [1, 2, 3]
    lm = {"4gram": gmm_4gram, "lstm": gmm_lstm}

    results = {}

    for lm, gmm_sys in lm.items():
        for n_phone in n_phones:
            with tk.block(f"{n_phone} {lm}"):
                results[n_phone, lm] = _run_hybrid(lm, n_phone, gmm_sys)

    return results
