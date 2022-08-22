from IPython import embed
import itertools
import os
import typing

# -------------------- Sisyphus --------------------
from sisyphus import gs, tk, Path

# -------------------- Recipes --------------------
from i6_core.meta import CartAndLDA
import i6_core.returnn as returnn
import i6_core.rasr as rasr

from i6_experiments.common.setups.rasr import GmmSystem
from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem
import i6_experiments.common.setups.rasr.util as rasr_util

from i6_private.users.gunz.setups.common.specaugment import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)

from .config_20_baseline_hybrid import get_hybrid_args, get_hybrid_system, get_lr_config


def get_returnn_config(
    num_inputs: int,
    num_outputs: int,
    num_epochs: int,
    blstm_size: int = 512,
    batch_size: int = 10000,
    lr: str = "v1",
) -> returnn.ReturnnConfig:
    base_config = {
        "use_tensorflow": True,
        "debug_print_layer_output_template": True,
        "log_batch_size": True,
        "tf_log_memory_usage": True,
        "cache_size": "0",
        "window": 1,
        "update_on_device": True,
        "extern_data": {
            "data": {"dim": num_inputs},
            "classes": {"dim": num_outputs, "sparse": True},
        },
        "batch_size": batch_size,
        "chunking": "100:50",
        "optimizer": {"class": "nadam"},
        "optimizer_epsilon": 1e-8,
        "gradient_noise": 0.1,
        **get_lr_config(num_epochs=num_epochs, lr_schedule=lr),
        "network": {
            "bwd_1": {
                "L2": 0.01,
                "class": "rec",
                "direction": -1,
                "dropout": 0.1,
                "from": "source",
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "bwd_2": {
                "L2": 0.01,
                "class": "rec",
                "direction": -1,
                "dropout": 0.1,
                "from": ["fwd_1", "bwd_1"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "bwd_3": {
                "L2": 0.01,
                "class": "rec",
                "direction": -1,
                "dropout": 0.1,
                "from": ["fwd_2", "bwd_2"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "bwd_4": {
                "L2": 0.01,
                "class": "rec",
                "direction": -1,
                "dropout": 0.1,
                "from": ["fwd_3", "bwd_3"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "bwd_5": {
                "L2": 0.01,
                "class": "rec",
                "direction": -1,
                "dropout": 0.1,
                "from": ["fwd_4", "bwd_4"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "bwd_6": {
                "L2": 0.01,
                "class": "rec",
                "direction": -1,
                "dropout": 0.1,
                "from": ["fwd_5", "bwd_5"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "output": {
                "class": "softmax",
                "from": "encoder-output",
                "loss": "ce",
                "loss_opts": {"focal_loss_factor": 2.0, "label_smoothing": 0.2},
                "target": "classes",
            },
            "encoder-output": {"class": "copy", "from": ["fwd_6", "bwd_6"]},
            "fwd_1": {
                "L2": 0.01,
                "class": "rec",
                "direction": 1,
                "dropout": 0.1,
                "from": "source",
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "fwd_2": {
                "L2": 0.01,
                "class": "rec",
                "direction": 1,
                "dropout": 0.1,
                "from": ["fwd_1", "bwd_1"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "fwd_3": {
                "L2": 0.01,
                "class": "rec",
                "direction": 1,
                "dropout": 0.1,
                "from": ["fwd_2", "bwd_2"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "fwd_4": {
                "L2": 0.01,
                "class": "rec",
                "direction": 1,
                "dropout": 0.1,
                "from": ["fwd_3", "bwd_3"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "fwd_5": {
                "L2": 0.01,
                "class": "rec",
                "direction": 1,
                "dropout": 0.1,
                "from": ["fwd_4", "bwd_4"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "fwd_6": {
                "L2": 0.01,
                "class": "rec",
                "direction": 1,
                "dropout": 0.1,
                "from": ["fwd_5", "bwd_5"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "source": {
                "class": "eval",
                "eval": "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)",
            },
        },
    }

    base_post_config = {
        "cleanup_old_models": {
            "keep_last_n": 5,
            "keep_best_n": 5,
            "keep": returnn.CodeWrapper(f"list(np.arange(10, {num_epochs + 1}, 10))"),
        },
    }

    returnn_cfg = returnn.ReturnnConfig(
        config=base_config,
        post_config=base_post_config,
        hash_full_python_code=True,
        python_prolog={
            "numpy": "import numpy as np",
            "functions": [
                sa_mask,
                sa_random_mask,
                sa_summary,
                sa_transform,
            ],
        },
        pprint_kwargs={"sort_dicts": False},
    )

    return returnn_cfg


def get_nn_args(
    *,
    gmm_system: GmmSystem,
    name: str,
    corpus_name: str,
    blstm_size: int,
    lr: str,
) -> rasr_util.HybridArgs:
    cart_job: CartAndLDA = gmm_system.jobs[corpus_name][
        f"cart_and_lda_{corpus_name}_mono"
    ]
    n_outputs = cart_job.last_num_cart_labels
    num_epochs = 500

    dict_cfg = get_returnn_config(
        num_inputs=50,
        num_outputs=n_outputs,
        num_epochs=num_epochs,
        blstm_size=blstm_size,
        batch_size=10000,
        lr=lr,
    )
    nn_args = get_hybrid_args(
        name=name, num_outputs=n_outputs, training_cfg=dict_cfg, fwd_cfg=dict_cfg
    )

    return nn_args


def run(returnn_root: tk.Path, gmm_4gram: GmmSystem) -> typing.Dict[str, HybridSystem]:
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    # ******************** HY Init ********************

    corpus_name = "train-other-960"
    layers = [6]
    lr = ["v1", "v2", "v3"]
    dim = [512]

    results = {}

    for lay, lr, dim in itertools.product(layers, lr, dim):
        name = f"blstm-ph:3-dim:{lay}x{dim}-lr:{lr}"
        with tk.block(name):
            system = get_hybrid_system(
                n_phones=3,
                gmm_system=gmm_4gram,
                corpus_name=corpus_name,
                returnn_root=returnn_root,
                diphone_state_tying_file=None,
            )
            nn_args = get_nn_args(
                gmm_system=gmm_4gram,
                name=name,
                corpus_name=corpus_name,
                blstm_size=512,
                lr=lr,
            )

            steps = rasr_util.RasrSteps()
            steps.add_step("nn", nn_args)
            system.run(steps)

            results[name] = system

    return results
