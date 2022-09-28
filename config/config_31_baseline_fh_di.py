import copy
import itertools
import numpy as np
import os
from IPython import embed

# -------------------- Sisyphus --------------------
from sisyphus import gs, tk, Path

# -------------------- Recipes --------------------

import i6_core.rasr as rasr
import i6_core.returnn as returnn

import i6_experiments.common.setups.rasr.util as rasr_util

from i6_private.users.gunz.setups.fh_ls.common.helpers.network_augment import (
    augment_net_with_label_pops,
    augment_net_with_monophone_outputs,
    augment_net_with_diphone_outputs,
)
import i6_private.users.gunz.setups.common.train_helpers as train_helpers
from i6_private.users.gunz.setups.common.specaugment import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)
import i6_private.users.gunz.setups.fh_ls.librispeech.factored_hybrid_system as fh_system
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
    RASR_ROOT_FH,
    RETURNN_PYTHON_ROSSENBACH_TF15,
    CONF_SIZES,
    CONF_NUM_HEADS,
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

    cfgs = itertools.product(CONF_SIZES, CONF_NUM_HEADS, [300])
    for conf_size, conf_num_heads, num_epochs in cfgs:
        run_(
            alignment=tk.Path(RAISSI_ALIGNMENT),
            returnn_root=returnn_root,
            conf_size=conf_size,
            conf_num_heads=conf_num_heads,
            num_epochs=num_epochs,
            lr="v4",
        )


def augment_net_with_diphone_outputs(network, use_multi_task):
    pass


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

    name = f"conf-ph:2-dim:{conf_size}-h:{conf_num_heads}-ep:{num_epochs}-lr:{lr}"
    print(f"fh {name}")

    # ***********Initial arguments and init step ********************
    (
        train_data_inputs,
        dev_data_inputs,
        test_data_inputs,
    ) = lbs_data_setups.get_data_inputs()
    rasr_init_args = lbs_data_setups.get_init_args(gt_normalization=False)
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
    s.label_info.state_tying = "no-tying-dense"
    s.label_info.use_boundary_classes = True  # Fair compairson w/ CART hybrid
    s.label_info.use_word_end_classes = False
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
        num_classes=s.label_info.get_n_of_dense_classes(),
        num_enc_layers=12,
        enc_args=encoder_args,
    )

    network = attention_for_hybrid(**network_args).get_network()
    network["encoder-output"] = {"class": "copy", "from": "encoder"}
    network = augment_net_with_label_pops(
        network,
        n_contexts=s.label_info.n_contexts,
        use_boundary_classes=True,
        use_word_end_classes=False,
    )
    network = augment_net_with_monophone_outputs(
        network,
        encoder_output_len=conf_size,
        add_mlps=True,
        final_ctx_type="triphone-forward",
        use_multi_task=False,
    )
    network = augment_net_with_diphone_outputs(
        network,
        encoder_output_len=conf_size,
        use_multi_task=False,
        ph_emb_size=s.label_info.ph_emb_size,
        st_emb_size=s.label_info.st_emb_size,
    )

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
        python_prolog={"numpy": "import numpy as np"},
        python_epilog={
            "dim_config": s.get_epilog_for_train(
                specaug_args=None  # we do SpecAug manually
            ),
            "functions": [
                sa_mask,
                sa_random_mask,
                sa_summary,
                sa_transform,
            ],
        },
    )

    s.set_experiment_dict("fh", "GMMtri", "di", postfix_name=name)
    s.experiments["fh"]["returnn_config"] = copy.deepcopy(returnn_config)

    train_args = {
        **s.initial_train_args,
        "returnn_config": returnn_config,
        "num_epochs": num_epochs,
        "partition_epochs": partition_epochs,
    }

    s.returnn_rasr_training(
        experiment_key="fh",
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
        nn_train_args=train_args,
    )

    s.set_diphone_priors(
        key="fh", epoch=num_epochs - (partition_epochs["train"] * 2) + 1, hdf_key=None
    )

    eval_epochs = list(np.arange(num_epochs // 2, num_epochs + 1, 50))

    for ep, crp_k in itertools.product(eval_epochs, ["dev-clean", "dev-other"]):
        recognizer, recog_args_mono = s.get_recognizer_and_args(
            key="fh",
            context_type=s.contexts["mono"],
            crp_corpus=crp_k,
            epoch=ep,
            num_encoder_output=conf_size,
            gpu=False,
        )

        del recog_args_mono["runOptJob"]

        args = itertools.product([18.0], [4.7, 5.0, 5.3], [0.6], [0.1], [20.0], [0.0])
        for beam, lm, t, p, exitSil, tdpExit in args:
            recog_cfg = {
                **recog_args_mono,
                "beam": beam,
                "lmScale": lm,
                "silExit": exitSil,
                "tdpExit": tdpExit,
                "tdpScale": t,
                "priorInfo": {
                    **recog_args_mono["priorInfo"],
                    "center-state-prior": {
                        **recog_args_mono["priorInfo"]["center-state-prior"],
                        "scale": p,
                    },
                },
            }

            recognizer.recognize_count_lm(**recog_cfg)
