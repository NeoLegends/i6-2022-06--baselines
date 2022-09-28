import itertools
import os
from IPython import embed

# -------------------- Sisyphus --------------------
from sisyphus import gs, tk, Path

# -------------------- Recipes --------------------

import i6_core.rasr as rasr

from .config_30_baseline_fh_mono import run_
from .config import (
    RASR_ROOT_FH,
    RETURNN_PYTHON_ROSSENBACH_TF15,
    CONF_SIZES,
    CONF_NUM_HEADS,
    CONF_NUM_TRAIN_EPOCHS,
    RAISSI_ALIGNMENT,
)

RASR_BINARY_PATH = tk.Path(os.path.join(RASR_ROOT_FH, "arch", gs.RASR_ARCH))
RASR_BINARY_PATH.hash_override = "FH_RASR_PATH"

RETURNN_PYTHON_EXE = tk.Path(RETURNN_PYTHON_ROSSENBACH_TF15)
RETURNN_PYTHON_EXE.hash_override = "FH_RETURNN_PYTHON_EXE"

train_key = "train-other-960"


def run(returnn_root: tk.Path):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    cfgs = itertools.product(CONF_SIZES, CONF_NUM_HEADS, CONF_NUM_TRAIN_EPOCHS)
    for conf_size, conf_num_heads, num_epochs in cfgs:
        run_(
            alignment=tk.Path(RAISSI_ALIGNMENT),
            alignment_name="GMMtri",
            returnn_root=returnn_root,
            conf_size=conf_size,
            conf_num_heads=conf_num_heads,
            num_epochs=num_epochs,
            lr="v4",
        )
