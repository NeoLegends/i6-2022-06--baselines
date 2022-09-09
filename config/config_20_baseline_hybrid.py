import copy
import itertools

from IPython import embed
import numpy as np
import os
import typing

# -------------------- Sisyphus --------------------
from sisyphus import gs, tk, Path

# -------------------- Recipes --------------------
import i6_core.corpus as corpus_recipe
from i6_core.lexicon import StoreAllophonesJob
from i6_core.meta import CartAndLDA, AlignSplitAccumulateSequence
import i6_core.returnn as returnn
import i6_core.rasr as rasr
import i6_core.text as text

from i6_experiments.common.setups.rasr import GmmSystem, ReturnnRasrDataInput
from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem
import i6_experiments.common.setups.rasr.util as rasr_util
from i6_experiments.users.luescher.helpers.search_params import get_search_parameters

import i6_private.users.gunz.setups.ls.pipeline_rasr_args as lbs_data_setups
import i6_private.users.gunz.setups.common.train_helpers as train_helpers
from i6_private.users.gunz.setups.common.specaugment import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)
from i6_private.users.gunz.system_librispeech.get_network_args import (
    get_encoder_args,
    get_network_args,
)
from i6_private.users.gunz.system_librispeech.transformer_network import (
    attention_for_hybrid,
)

from .config import (
    N_PHONES,
    RAISSI_ALIGNMENT,
    RASR_ROOT_2019,
    RETURNN_PYTHON_ROSSENBACH_TF15,
)

RASR_BINRARY_PATH = tk.Path(os.path.join(RASR_ROOT_2019, "arch", gs.RASR_ARCH))
RASR_BINRARY_PATH.hash_override = "LS_RASR_PATH"

RETURNN_PYTHON_EXE = tk.Path(RETURNN_PYTHON_ROSSENBACH_TF15)
RETURNN_PYTHON_EXE.hash_override = "LS_RETURNN_PYTHON_EXE"


def n_phones_to_str(n_phones: int) -> str:
    if n_phones == 1:
        return "mono"
    elif n_phones == 2:
        return "di"
    elif n_phones == 3:
        return "tri"
    else:
        raise ValueError(f"n_phones must be either 1, 2 or 3, not {n_phones}")


def get_lr_config(num_epochs: int, lr_schedule: str = "v1"):
    base = {
        "learning_rate_file": "lr.log",
        "min_learning_rate": 1e-6,
    }

    if lr_schedule == "v1":
        # Taken from Chris Lüscher

        return {
            **base,
            "learning_rates": list(np.linspace(3e-4, 8e-4, 10)),
            "learning_rate_control": "newbob_multi_epoch",
            "learning_rate_control_min_num_epochs_per_new_lr": 3,
            "learning_rate_control_relative_error_relative_lr": True,
            "newbob_learning_rate_decay": 0.9,
            "newbob_multi_num_epochs": 40,
            "newbob_multi_update_interval": 1,
        }
    elif lr_schedule == "v2":
        # OneCycle: https://www-i6.informatik.rwth-aachen.de/publications/download/1204/Zhou--2022.pdf

        lr_peak = 1e-4
        rates = (
            list(np.linspace(lr_peak / 10, lr_peak, int(num_epochs * 0.45)))
            + list(np.linspace(lr_peak, lr_peak / 10, int(num_epochs * 0.45)))
            + [1e-6]
        )
        return {
            **base,
            "learning_rates": rates,
            "learning_rate_control": "constant",
        }
    elif lr_schedule == "v3":
        # OneCycle from Wei

        n = int(num_epochs * 0.45)
        schedule = train_helpers.get_learning_rates(lrate=5e-5, increase=n, decay=n)

        return {
            **base,
            "learning_rates": schedule,
            "learning_rate_control": "constant",
        }
    elif lr_schedule == "v4":
        # OneCycle from Wei

        n = int(num_epochs * 0.45)
        schedule = train_helpers.get_learning_rates(increase=n, decay=n)

        return {
            **base,
            "learning_rates": schedule,
            "learning_rate_control": "constant",
        }
    else:
        raise ValueError(f"unknown LR {lr_schedule}")


def get_returnn_config(
    num_inputs: int,
    num_outputs: int,
    num_epochs: int,
    conf_size: int,
    conf_num_heads: int,
    batch_size: int = 10000,
    lr: str = "v1",
) -> returnn.ReturnnConfig:
    encoder_args = get_encoder_args(
        num_heads=conf_num_heads,
        key_dim_per_head=int(conf_size / conf_num_heads),
        value_dim_per_head=int(conf_size / conf_num_heads),
        model_dim=conf_size,
        ff_dim=int(conf_size * 4),
        kernel_size=32,
    )
    network_args = get_network_args(
        type="conformer",
        num_classes=num_outputs,
        num_enc_layers=12,
        enc_args=encoder_args,
    )
    network = attention_for_hybrid(**network_args).get_network()

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
        "network": network,
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


def get_hybrid_args(
    name: str,
    num_outputs: int,
    training_cfg: returnn.ReturnnConfig,
    fwd_cfg: returnn.ReturnnConfig,
    num_epochs: int,
):
    training_args = {
        "log_verbosity": 4,
        "num_epochs": num_epochs,
        "num_classes": num_outputs,
        "save_interval": 1,
        "keep_epochs": None,
        "time_rqmt": 168,
        "mem_rqmt": 7,
        "cpu_rqmt": 3,
        "partition_epochs": {"train": 20, "dev": 1},
        "use_python_control": False,
    }
    recognition_args = {
        "dev-other": {
            "epochs": list(np.arange(num_epochs // 2, num_epochs + 1, 50)),
            "feature_flow_key": "gt",
            "prior_scales": [0.3],
            "pronunciation_scales": [6.0],
            "lm_scales": [20.0],
            "lm_lookahead": True,
            "lookahead_options": None,
            "create_lattice": True,
            "eval_single_best": True,
            "eval_best_in_lattice": True,
            "search_parameters": get_search_parameters(),
            "lattice_to_ctm_kwargs": {
                "fill_empty_segments": True,
                "best_path_algo": "bellman-ford",
            },
            "optimize_am_lm_scale": False,
            "rtf": 50,
            "mem": 16,
            "use_gpu": True,
            "parallelize_conversion": True,
        },
    }
    test_recognition_args = None

    nn_args = rasr_util.HybridArgs(
        returnn_training_configs={name: training_cfg},
        returnn_recognition_configs={name: fwd_cfg},
        training_args=training_args,
        recognition_args=recognition_args,
        test_recognition_args=test_recognition_args,
    )

    return nn_args


def get_nn_args(
    *,
    gmm_system: GmmSystem,
    name: str,
    corpus_name: str,
    conf_size: int,
    conf_num_heads: int,
    n_phones: int,
    lr: str,
    num_epochs: int,
) -> rasr_util.HybridArgs:
    if n_phones == 1:
        allophones_job: StoreAllophonesJob = gmm_system.jobs["train-other-960"][
            "allophones"
        ]
        n_outputs = allophones_job.out_num_monophone_states
    elif n_phones == 2:
        cart_lda: CartAndLDA = gmm_system.jobs["train-other-960"][
            "cart_and_lda_train-other-960_cart_di"
        ]
        n_outputs = cart_lda.last_num_cart_labels
    else:
        cart_job: CartAndLDA = gmm_system.jobs[corpus_name][
            f"cart_and_lda_{corpus_name}_mono"
        ]
        n_outputs = cart_job.last_num_cart_labels

    batch_sizes = {
        (256, 8): 8192,
        (256, 12): 10000,
        (256, 32): 10000,
        (384, 8): 6144,
        (384, 12): 10000,
        (384, 32): 4096,
        (512, 8): 4096,
        (512, 12): 4096,
        (512, 32): 3584,
        (768, 12): 2048,
        (786, 32): 2048,
    }
    batch_size = batch_sizes.get((conf_size, conf_num_heads), min(batch_sizes.values()))

    dict_cfg = get_returnn_config(
        num_inputs=50,
        num_outputs=n_outputs,
        num_epochs=num_epochs,
        conf_size=conf_size,
        conf_num_heads=conf_num_heads,
        batch_size=batch_size,
        lr=lr,
    )
    fwd_config = copy.deepcopy(dict_cfg)
    fwd_config.config["forward_output_layer"] = "output"
    nn_args = get_hybrid_args(
        name=name,
        num_outputs=n_outputs,
        training_cfg=dict_cfg,
        fwd_cfg=fwd_config,
        num_epochs=num_epochs,
    )

    return nn_args


def get_hybrid_system(
    *,
    corpus_name: str,
    n_phones: int,
    gmm_system: GmmSystem,
    returnn_root: tk.Path,
) -> HybridSystem:
    assert n_phones in [1, 2, 3]

    # ******************** Data Prep ********************

    train_corpus_path = gmm_system.corpora[corpus_name].corpus_file
    total_train_num_segments = 281241
    cv_size = 3000 / total_train_num_segments

    all_segments = corpus_recipe.SegmentCorpusJob(
        train_corpus_path, 1
    ).out_single_segment_files[1]

    splitted_segments_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
        all_segments, {"train": 1 - cv_size, "cv": cv_size}
    )
    train_segments = splitted_segments_job.out_segments["train"]
    cv_segments = splitted_segments_job.out_segments["cv"]
    devtrain_segments = text.TailJob(
        train_segments, num_lines=1000, zip_output=False
    ).out

    # ******************** Train Prep ********************

    output_name = "tri" if n_phones == 3 else "di" if n_phones == 2 else "mono"
    train_output = gmm_system.outputs[corpus_name][output_name]

    nn_train_data: ReturnnRasrDataInput = train_output.as_returnn_rasr_data_input(
        shuffle_data=True
    )
    nn_train_data.update_crp_with(segment_path=train_segments, concurrent=1)

    nn_cv_data: ReturnnRasrDataInput = train_output.as_returnn_rasr_data_input()
    nn_cv_data.update_crp_with(segment_path=cv_segments, concurrent=1)

    nn_devtrain_data: ReturnnRasrDataInput = train_output.as_returnn_rasr_data_input()
    nn_devtrain_data.update_crp_with(segment_path=devtrain_segments, concurrent=1)

    if n_phones == 1:
        alignment_job: AlignSplitAccumulateSequence = gmm_system.jobs[corpus_name][
            "train_mono"
        ]
        align = alignment_job.selected_alignment_jobs[-1].out_alignment_bundle

        nn_train_data.crp.acoustic_model_config.state_tying.type = "monophone"
        nn_devtrain_data.crp.acoustic_model_config.state_tying.type = "monophone"
        nn_cv_data.crp.acoustic_model_config.state_tying.type = "monophone"
    elif n_phones == 2:
        cart_lda: CartAndLDA = gmm_system.jobs["train-other-960"][
            "cart_and_lda_train-other-960_cart_di"
        ]

        nn_train_data.crp.acoustic_model_config.state_tying.file = (
            cart_lda.last_cart_tree
        )
        nn_devtrain_data.crp.acoustic_model_config.state_tying.file = (
            cart_lda.last_cart_tree
        )
        nn_cv_data.crp.acoustic_model_config.state_tying.file = cart_lda.last_cart_tree

        # use Raissi triphone alignment (for now)
        align = tk.Path(RAISSI_ALIGNMENT)
    else:
        # use Raissi triphone alignment (for now)
        align = tk.Path(RAISSI_ALIGNMENT)

    nn_train_data.alignments = align
    nn_devtrain_data.alignments = align
    nn_cv_data.alignments = align

    nn_train_data_inputs = {
        f"{corpus_name}.train": nn_train_data,
    }
    nn_cv_data_inputs = {
        f"{corpus_name}.cv": nn_cv_data,
    }
    nn_devtrain_data_inputs = {
        f"{corpus_name}.devtrain": nn_devtrain_data,
    }

    # ******************** Test Prep ********************

    nn_dev_data_inputs = {
        # "dev-clean": lbs_gmm_system.outputs["dev-clean"][
        #    "final"
        # ].as_returnn_rasr_data_input(),
        "dev-other.dev": gmm_system.outputs["dev-other"][
            output_name
        ].as_returnn_rasr_data_input(),
    }
    nn_test_data_inputs = {
        # "test-clean": lbs_gmm_system.outputs["test-clean"][
        #    "final"
        # ].as_returnn_rasr_data_input(),
        "test-other.test": gmm_system.outputs["test-other"][
            output_name
        ].as_returnn_rasr_data_input(),
    }

    # ******************** System Init ********************

    hybrid_init_args = lbs_data_setups.get_init_args()

    lbs_hy_system = HybridSystem(
        rasr_binary_path=RASR_BINRARY_PATH,
        returnn_root=returnn_root,
        returnn_python_exe=RETURNN_PYTHON_EXE,
    )
    lbs_hy_system.init_system(
        rasr_init_args=hybrid_init_args,
        train_data=nn_train_data_inputs,
        cv_data=nn_cv_data_inputs,
        devtrain_data=nn_devtrain_data_inputs,
        dev_data=nn_dev_data_inputs,
        test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple([f"{corpus_name}.train", f"{corpus_name}.cv"])],
    )

    for corpus in ["dev-other.dev", "test-other.test"]:
        lbs_hy_system.crp[
            corpus
        ].acoustic_model_config.mixture_set.allow_zero_weights = True
        lbs_hy_system.crp[
            corpus
        ].acoustic_model_config.old_mixture_set.allow_zero_weights = True

    return lbs_hy_system


def run(
    returnn_root: tk.Path,
    gmm_4gram: GmmSystem,
) -> typing.Dict[str, HybridSystem]:
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    # ******************** HY Init ********************

    corpus_name = "train-other-960"
    lm = {"4gram": gmm_4gram}  # , "lstm": gmm_lstm}
    lr = ["v4"]
    num_heads = [8]
    sizes = [256, 384, 512]
    num_epochs = [300, 400, 500]

    results = {}

    for (
        (lm, gmm_sys),
        n_phone,
        conf_size,
        conf_num_heads,
        lr,
        num_epochs,
    ) in itertools.product(lm.items(), N_PHONES, sizes, num_heads, lr, num_epochs):
        if conf_size % conf_num_heads != 0:
            print(f"{conf_size} does not work w/ {conf_num_heads} att heads, skipping")
            continue

        name = f"conf-ph:{n_phone}-dim:{conf_size}-h:{conf_num_heads}-ep:{num_epochs}-lr:{lr}"

        with tk.block(name):
            print(f"hy {name}")
            system = get_hybrid_system(
                n_phones=n_phone,
                gmm_system=gmm_sys,
                corpus_name=corpus_name,
                returnn_root=returnn_root,
            )
            nn_args = get_nn_args(
                gmm_system=gmm_sys,
                name=name,
                corpus_name=corpus_name,
                conf_size=conf_size,
                conf_num_heads=conf_num_heads,
                n_phones=n_phone,
                lr=lr,
                num_epochs=num_epochs,
            )

            steps = rasr_util.RasrSteps()
            steps.add_step("nn", nn_args)
            system.run(steps)

            results[name] = system

    return results
