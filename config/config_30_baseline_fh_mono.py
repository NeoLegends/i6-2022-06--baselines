import copy
import itertools
import os
from IPython import embed

# -------------------- Sisyphus --------------------
from sisyphus import gs, tk, Path

# -------------------- Recipes --------------------

from i6_core.meta import AlignSplitAccumulateSequence
import i6_core.rasr as rasr
import i6_core.returnn as returnn

from i6_experiments.common.setups.rasr import GmmSystem
import i6_experiments.common.setups.rasr.util as rasr_util
import i6_experiments.users.raissi.setups.librispeech.factored_hybrid_system as fh_system

import i6_private.users.gunz.setups.common.train_helpers as train_helpers
from i6_private.users.gunz.setups.common.specaugment import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)
import i6_private.users.gunz.setups.ls.pipeline_gmm_args as gmm_setups
import i6_private.users.gunz.setups.ls.pipeline_rasr_args as lbs_data_setups
import i6_private.users.gunz.system_librispeech.batch_sizes as batch_sizes
from i6_private.users.gunz.system_librispeech.get_network_args import (
    get_encoder_args,
    get_network_args,
)
from i6_private.users.gunz.system_librispeech.transformer_network import (
    attention_for_hybrid,
)

from .config import (
    RASR_ROOT_2019,
    RETURNN_PYTHON_ROSSENBACH_TF15,
    CONF_SIZES,
    CONF_NUM_HEADS,
    CONF_NUM_TRAIN_EPOCHS,
)

RASR_BINARY_PATH = tk.Path(os.path.join(RASR_ROOT_2019, "arch", gs.RASR_ARCH))
RASR_BINARY_PATH.hash_override = "LS_RASR_PATH"

RETURNN_PYTHON_EXE = tk.Path(RETURNN_PYTHON_ROSSENBACH_TF15)
RETURNN_PYTHON_EXE.hash_override = "LS_RETURNN_PYTHON_EXE"

train_key = "train-other-960"


def run(gmm: GmmSystem, returnn_root: tk.Path):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    alignment_job: AlignSplitAccumulateSequence = gmm.jobs[train_key]["train_mono"]
    align = alignment_job.selected_alignment_jobs[-1].out_alignment_bundle

    cfgs = itertools.product(CONF_SIZES, CONF_NUM_HEADS, CONF_NUM_TRAIN_EPOCHS)
    for conf_size, conf_num_heads, num_epochs in cfgs:
        run_(
            alignment=align,
            returnn_root=returnn_root,
            conf_size=conf_size,
            conf_num_heads=conf_num_heads,
            num_epochs=num_epochs,
            lr="v4",
        )


def run_(
    *,
    alignment: tk.Path,
    returnn_root: tk.Path,
    conf_size: int,
    conf_num_heads: int,
    num_epochs: int,
    lr: str,
):
    # ******************** HY Init ********************

    name = f"conf-ph:1-dim:{conf_size}-h:{conf_num_heads}-ep:{num_epochs}-lr:{lr}"
    print(f"fh {name}")

    # ***********Initial arguments and init step ********************
    (
        train_data_inputs,
        dev_data_inputs,
        test_data_inputs,
    ) = lbs_data_setups.get_data_inputs()
    rasr_init_args = lbs_data_setups.get_init_args(dc_detection=False)
    data_preparation_args = gmm_setups.get_final_output(name="data_preparation")
    # *********** System Instantiation *****************
    steps = rasr_util.RasrSteps()
    steps.add_step("init", None)  # you can create the label_info and pass here
    s = fh_system.FactoredHybridSystem(
        rasr_binary_path=RASR_BINARY_PATH,
        rasr_init_args=rasr_init_args,
        returnn_root=returnn_root,
        returnn_python_exe=RETURNN_PYTHON_EXE,
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )
    s.train_key = train_key
    s.label_info.use_word_end_classes = True
    s.run(steps)

    # *********** Preparation of data input for rasr-returnn training *****************
    s.alignments[train_key] = alignment
    steps_input = rasr_util.RasrSteps()
    steps_input.add_step("extract", rasr_init_args.feature_extraction_args)
    steps_input.add_step("input", data_preparation_args)
    s.run(steps_input)

    s.set_crp_pairings()
    s.set_rasr_returnn_input_datas(
        is_cv_separate_from_train=False,
        input_key="data_preparation",
        chunk_size="100:50",
    )
    s._update_am_setting_for_all_crps(
        train_tdp_type="default",
        eval_tdp_type="default",
    )

    # ---------------------- returnn config---------------
    partition_epochs = {"train": 20, "dev": 1}
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
        num_classes=s.label_info.get_n_state_classes(),
        num_enc_layers=12,
        enc_args=encoder_args,
    )
    network = attention_for_hybrid(**network_args).get_network()
    network["center-output"] = network.pop("output")

    base_config = {
        **s.initial_nn_args,
        **train_helpers.get_returnn_lr_config(num_epochs=num_epochs),
        "batch_size": batch_sizes.get_conf_batch_size(conf_size, conf_num_heads),
        "use_tensorflow": True,
        "debug_print_layer_output_template": True,
        "log_batch_size": True,
        "tf_log_memory_usage": True,
        "cache_size": "0",
        "window": 1,
        "update_on_device": True,
        "chunking": "100:50",
        "optimizer": {"class": "nadam"},
        "optimizer_epsilon": 1e-8,
        "gradient_noise": 0.1,
        "network": network,
    }
    base_post_config = {
        "cleanup_old_models": {
            "keep_last_n": 5,
            "keep_best_n": 5,
            "keep": returnn.CodeWrapper(f"list(np.arange(10, {num_epochs + 1}, 10))"),
        },
    }
    returnn_config = returnn.ReturnnConfig(
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
    )

    s.set_experiment_dict("fh", "GMMmono", "mono", postfix_name=name)

    train_args = {
        **s.initial_train_args,
        "returnn_config": returnn_config,
        "num_epochs": num_epochs,
        "partition_epochs": partition_epochs,
    }

    s.returnn_rasr_training(
        experiment_key=name,
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
        nn_train_args=train_args,
    )

    """
    s.set_mono_priors(
        key=name, epoch=num_epochs - (partition_epochs["train"] * 2) + 1, hdf_key=None
    )

    if epochs is not None:
        for e in epochs:
            for crp_k in ["dev-other"]:
                recognizer, recog_args_mono = s.get_recognizer_and_args(
                    key=key,
                    context_type=s.contexts["mono"],
                    crp_corpus=crp_k,
                    epoch=e,
                )

                for beam in [18.0]:
                    recog_args_mono["beam"] = beam
                    recog_args_mono["lmScale"] = kw["lm"]
                    for t in [kw["t"]]:
                        for p in [kw["p"]]:
                            for exitSil in [20.0]:
                                for tdpExit in [0.0]:
                                    recog_args_mono["silExit"] = exitSil
                                    recog_args_mono["tdpExit"] = tdpExit
                                    recog_args_mono["tdpScale"] = t
                                    recog_args_mono["priorInfo"]["center-state-prior"][
                                        "scale"
                                    ] = p
                                    recognizer.recognize_count_lm(**recog_args_mono)
    """
