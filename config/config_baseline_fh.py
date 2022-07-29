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
import i6_experiments.users.luescher.setups.librispeech.pipeline_base_args as lbs_setups
import i6_experiments.users.raissi.experiments.librispeech.data_preparation.clean_100h.data as data_setups


def run_fh(gmm_4gram: GmmSystem, gmm_lstm: GmmSystem):
    print("FH")

    pass
