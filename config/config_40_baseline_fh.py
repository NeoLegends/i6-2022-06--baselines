import os
from IPython import embed

# -------------------- Sisyphus --------------------

from sisyphus import gs, tk, Path

# -------------------- Recipes --------------------

import i6_core.rasr as rasr

from i6_core.tools import CloneGitRepositoryJob

from i6_experiments.common.setups.rasr.gmm_system import GmmSystem
import i6_experiments.common.setups.rasr.nn_system as nn_system
import i6_experiments.common.setups.rasr.util as rasr_util


def run_fh(gmm_4gram: GmmSystem, gmm_lstm: GmmSystem):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    # ******************** HY Init ********************

    print("FH")
    pass
