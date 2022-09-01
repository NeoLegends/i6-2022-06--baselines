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
            "lstm_bwd_1": {
                "L2": 0.01,
                "class": "rec",
                "direction": -1,
                "dropout": 0.1,
                "from": ["data"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "lstm_bwd_2": {
                "L2": 0.01,
                "class": "rec",
                "direction": -1,
                "dropout": 0.1,
                "from": ["lstm_fwd_1", "lstm_bwd_1"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "lstm_bwd_3": {
                "L2": 0.01,
                "class": "rec",
                "direction": -1,
                "dropout": 0.1,
                "from": ["lstm_fwd_2", "lstm_bwd_2"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "lstm_bwd_4": {
                "L2": 0.01,
                "class": "rec",
                "direction": -1,
                "dropout": 0.1,
                "from": ["lstm_fwd_3", "lstm_bwd_3"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "lstm_bwd_5": {
                "L2": 0.01,
                "class": "rec",
                "direction": -1,
                "dropout": 0.1,
                "from": ["lstm_fwd_4", "lstm_bwd_4"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "lstm_bwd_6": {
                "L2": 0.01,
                "class": "rec",
                "direction": -1,
                "dropout": 0.1,
                "from": ["lstm_fwd_5", "lstm_bwd_5"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "lstm_fwd_1": {
                "L2": 0.01,
                "class": "rec",
                "direction": 1,
                "dropout": 0.1,
                "from": ["data"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "lstm_fwd_2": {
                "L2": 0.01,
                "class": "rec",
                "direction": 1,
                "dropout": 0.1,
                "from": ["lstm_fwd_1", "lstm_bwd_1"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "lstm_fwd_3": {
                "L2": 0.01,
                "class": "rec",
                "direction": 1,
                "dropout": 0.1,
                "from": ["lstm_fwd_2", "lstm_bwd_2"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "lstm_fwd_4": {
                "L2": 0.01,
                "class": "rec",
                "direction": 1,
                "dropout": 0.1,
                "from": ["lstm_fwd_3", "lstm_bwd_3"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "lstm_fwd_5": {
                "L2": 0.01,
                "class": "rec",
                "direction": 1,
                "dropout": 0.1,
                "from": ["lstm_fwd_4", "lstm_bwd_4"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "lstm_fwd_6": {
                "L2": 0.01,
                "class": "rec",
                "direction": 1,
                "dropout": 0.1,
                "from": ["lstm_fwd_5", "lstm_bwd_5"],
                "n_out": blstm_size,
                "unit": "nativelstm2",
            },
            "output": {
                "class": "softmax",
                "from": ["lstm_fwd_6", "lstm_bwd_6"],
                "loss": "ce",
                "loss_opts": {"focal_loss_factor": 2.0, "label_smoothing": 0.2},
                "target": "classes",
                "n_out": num_outputs,
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

    base_output_folder = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    # ******************** HY Init ********************

    corpus_name = "train-other-960"
    layers = [6]
    lr = ["v2", "v4"]
    dim = [512]

    results = {}

    for lay, lr, dim in itertools.product(layers, lr, dim):
        name = f"blstm-ph:3-dim:{lay}x{dim}-lr:{lr}"
        print(name)

        gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.join(base_output_folder, name)

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
