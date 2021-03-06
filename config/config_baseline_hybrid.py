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
import i6_core.tools as tools

from i6_experiments.common.setups.returnn_common import serialization
from i6_experiments.common.setups.rasr import GmmSystem, ReturnnRasrDataInput
from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem
import i6_experiments.common.setups.rasr.util as rasr_util
from i6_experiments.users.luescher.helpers.search_params import get_search_parameters

import i6_private.users.gunz.rasr.ls_base_args as lbs_data_setups
from i6_private.users.gunz.system_librispeech.get_network_args import (
    get_encoder_args,
    get_network_args,
)
from i6_private.users.gunz.system_librispeech.specaugment_new import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)
from i6_private.users.gunz.system_librispeech.transformer_network import (
    attention_for_hybrid,
)

from .config import N_PHONES, RAISSI_ALIGNMENT


def n_phones_to_str(n_phones: int) -> str:
    if n_phones == 1:
        return "mono"
    elif n_phones == 2:
        return "di"
    elif n_phones == 3:
        return "tri"
    else:
        raise ValueError(f"n_phones must be either 1, 2 or 3, not {n_phones}")


def get_returnn_config(
    num_inputs: int,
    num_outputs: int,
    num_epochs: int,
    conf_size: int,
    batch_size: int = 10000,
) -> returnn.ReturnnConfig:
    num_heads = 8
    encoder_args = get_encoder_args(
        num_heads,
        int(conf_size / num_heads),
        int(conf_size / num_heads),
        conf_size,
        int(conf_size * 4),
        32,
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
        "learning_rates": returnn.CodeWrapper("list(np.linspace(3e-4, 8e-4, 10))"),
        "learning_rate_control": "newbob_multi_epoch",
        "learning_rate_control_min_num_epochs_per_new_lr": 3,
        "learning_rate_control_relative_error_relative_lr": True,
        "min_learning_rate": 1e-5,
        "newbob_learning_rate_decay": 0.9,
        "newbob_multi_num_epochs": 40,
        "newbob_multi_update_interval": 1,
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


def get_returnn_common_args(
    num_inputs: int,
    num_outputs: int,
    num_epochs: int,
    conf_size: int,
    training: bool,
    returnn_common_root: tk.Path,
    batch_size: int = 12500,
) -> returnn.ReturnnConfig:
    config = {
        "behavior_version": 12,
        ############
        "optimizer": {"class": "adam", "epsilon": 1e-8},
        "update_on_device": True,
        "batch_size": batch_size,
        "chunking": "100:50",
        "optimizer_epsilon": 1e-8,
        "gradient_noise": 0.1,
        "window": 1,
        ############
        "learning_rates": returnn.CodeWrapper("list(np.linspace(3e-4, 8e-4, 10))"),
        "learning_rate_control": "newbob_multi_epoch",
        "learning_rate_control_min_num_epochs_per_new_lr": 3,
        "learning_rate_control_relative_error_relative_lr": True,
        "min_learning_rate": 1e-5,
        ############
        "newbob_learning_rate_decay": 0.9,
        "newbob_multi_num_epochs": 40,
        "newbob_multi_update_interval": 1,
    }
    post_config = {
        "use_tensorflow": True,
        "tf_log_memory_usage": True,
        "stop_on_nonfinite_train_score": True,
        "log_batch_size": True,
        "debug_print_layer_output_template": True,
        "cache_size": "0",
        "cleanup_old_models": {
            "keep_last_n": 5,
            "keep_best_n": 5,
            "keep": returnn.CodeWrapper(f"list(np.arange(10, {num_epochs + 1}, 10))"),
        },
    }

    data_dim = serialization.DataInitArgs(
        name="data",
        dim_tags=[
            serialization.DimInitArgs(name="data_time", dim=None),
            serialization.DimInitArgs(
                name="data_feature", dim=num_inputs, is_feature=True
            ),
        ],
        sparse_dim=None,
        available_for_inference=True,
    )
    classes_dim = serialization.DataInitArgs(
        name="classes",
        dim_tags=[serialization.DimInitArgs(name="classes_time", dim=None)],
        sparse_dim=serialization.DimInitArgs(
            name="classes_idx", dim=num_outputs, is_feature=True
        ),
        available_for_inference=False,
    )

    rc_recursion_limit = serialization.PythonEnlargeStackWorkaroundCode
    rc_extern_data = serialization.ExternData(extern_data=[data_dim, classes_dim])
    model_base = "i6_private.users.gunz.conformer"
    rc_model = serialization.Import(f"{model_base}.Conformer")
    rc_create_model = serialization.Import(f"{model_base}.network.create_network")
    rc_network = serialization.Network(
        rc_create_model.object_name,
        {
            "spatial_dim": "data_time",
            "features_data": "data",
            "target_data": "classes",
        },
        {
            "num_blocks": 12,
            "model_dim": conf_size,
            "out_dim": num_outputs,
            "training": training,
        },
    )
    rc_serializer = serialization.Collection(
        make_local_package_copy=False,
        packages={model_base},
        returnn_common_root=returnn_common_root,
        serializer_objects=[
            rc_recursion_limit,
            rc_extern_data,
            rc_model,
            rc_create_model,
            rc_network,
        ],
    )

    cfg = returnn.ReturnnConfig(
        config=config,
        post_config=post_config,
        python_epilog=[
            rc_serializer,
            "network = get_network(epoch=0)" if not training else "",
        ],
        python_prolog=[
            "import numpy as np",
        ],
    )
    return cfg


def get_hybrid_args(
    num_outputs: int,
    training_cfg: returnn.ReturnnConfig,
    fwd_cfg: returnn.ReturnnConfig,
    num_epochs: int = 500,
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
            "epochs": list(np.arange(250, num_epochs + 1, 10)),
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
            "mem": 8,
            "parallelize_conversion": True,
        },
    }
    test_recognition_args = None

    nn_args = rasr_util.HybridArgs(
        returnn_training_configs={"conf": training_cfg},
        returnn_recognition_configs={"conf": fwd_cfg},
        training_args=training_args,
        recognition_args=recognition_args,
        test_recognition_args=test_recognition_args,
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

    train_output = gmm_system.outputs[corpus_name]["final"]

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

        nn_train_data.crp.acoustic_model_config.state_tying.type = (
            nn_devtrain_data.crp.acoustic_model_config.state_tying.type
        ) = nn_cv_data.crp.acoustic_model_config.state_tying.type = "monophone"
    elif n_phones == 2:
        raise NotImplementedError("diphones not supported yet")
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
            "final"
        ].as_returnn_rasr_data_input(),
    }
    nn_test_data_inputs = {
        # "test-clean": lbs_gmm_system.outputs["test-clean"][
        #    "final"
        # ].as_returnn_rasr_data_input(),
        "test-other.test": gmm_system.outputs["test-other"][
            "final"
        ].as_returnn_rasr_data_input(),
    }

    # ******************** System Init ********************

    hybrid_init_args = lbs_data_setups.get_init_args(
        am_extra_args={
            "states_per_phone": n_phones,
        }
    )

    rasr_path = os.path.join(gs.RASR_ROOT, "arch", gs.RASR_ARCH)
    lbs_hy_system = HybridSystem(
        rasr_binary_path=tk.Path(rasr_path), returnn_root=returnn_root
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

    return lbs_hy_system


def get_nn_args(
    *,
    gmm_system: GmmSystem,
    corpus_name: str,
    conf_size: int,
    n_phones: int,
    returnn_common_root: typing.Optional[tk.Path] = None,
) -> rasr_util.HybridArgs:
    if n_phones == 1:
        allophones_job: StoreAllophonesJob = gmm_system.jobs["train-other-960"][
            "allophones"
        ]
        n_outputs = allophones_job.out_num_monophone_states
    elif n_phones == 2:
        raise NotImplementedError("diphones not supported yet")
    else:
        cart_job: CartAndLDA = gmm_system.jobs[corpus_name][
            f"cart_and_lda_{corpus_name}_mono"
        ]
        n_outputs = cart_job.last_num_cart_labels

    num_epochs = 500
    batch_size = 4096 if conf_size > 256 else 10000

    if returnn_common_root:
        returnn_common_training_config = get_returnn_common_args(
            num_inputs=50,
            num_outputs=int(n_outputs.get()),
            num_epochs=num_epochs,
            conf_size=conf_size,
            training=True,
            batch_size=batch_size,
            returnn_common_root=returnn_common_root,
        )
        returnn_common_fwd_config = get_returnn_common_args(
            num_inputs=50,
            num_outputs=int(n_outputs.get()),
            num_epochs=num_epochs,
            conf_size=conf_size,
            training=False,
            batch_size=batch_size,
            returnn_common_root=returnn_common_root,
        )
        nn_args = get_hybrid_args(
            num_outputs=int(n_outputs.get()),
            training_cfg=returnn_common_training_config,
            fwd_cfg=returnn_common_fwd_config,
        )
    else:
        dict_cfg = get_returnn_config(
            num_inputs=50,
            num_outputs=n_outputs,
            num_epochs=num_epochs,
            conf_size=conf_size,
            batch_size=batch_size,
        )
        nn_args = get_hybrid_args(
            num_outputs=int(n_outputs.get()), training_cfg=dict_cfg, fwd_cfg=dict_cfg
        )

    return nn_args


def run_hybrid(
    gmm_4gram: GmmSystem, gmm_lstm: GmmSystem
) -> typing.Dict[typing.Tuple[int, str], HybridSystem]:
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    # ******************** get returnn_common ********************

    clone_r_job = tools.CloneGitRepositoryJob(
        url="https://github.com/rwth-i6/returnn.git",
        commit="aadac2637ed6ec00925b9debf0dbd3c0ee20d6a6",
        checkout_folder_name="returnn",
    )
    clone_rc_job = tools.CloneGitRepositoryJob(
        url="https://github.com/rwth-i6/returnn_common.git",
        commit="79876b18552f61a3af7c21c670475fee51ef3991",
        checkout_folder_name="returnn_common",
    )

    # ******************** HY Init ********************

    lm = {"4gram": gmm_4gram}  # , "lstm": gmm_lstm}
    sizes = [256, 512]
    use_returnn_common_cfg = [True]
    corpus_name = "train-other-960"

    results = {}

    for (lm, gmm_sys), n_phone, conf_size, use_returnn_common in itertools.product(
        lm.items(), N_PHONES, sizes, use_returnn_common_cfg
    ):
        rc = "rc" if use_returnn_common else "r"
        name = f"hy {n_phones_to_str(n_phone)} {lm} {conf_size} {rc}"
        print(name)

        with tk.block(name):
            system = get_hybrid_system(
                n_phones=n_phone,
                gmm_system=gmm_sys,
                corpus_name=corpus_name,
                returnn_root=clone_r_job.out_repository,
            )
            nn_args = get_nn_args(
                gmm_system=gmm_sys,
                corpus_name=corpus_name,
                conf_size=conf_size,
                n_phones=n_phone,
                returnn_common_root=clone_rc_job.out_repository
                if use_returnn_common
                else None,
            )

            steps = rasr_util.RasrSteps()
            steps.add_step("nn", nn_args)
            system.run(steps)

            results[n_phone, lm] = system

    return results
