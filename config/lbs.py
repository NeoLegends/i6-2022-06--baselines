import copy

import i6_experiments.common.datasets.librispeech as lbs_dataset
import i6_private.users.gunz.setups.ls.pipeline_rasr_args as data_setups


def get_data_4gram_lm():
    return data_setups.get_data_inputs(add_unknown_phoneme_and_mapping=True)


def get_data_lstm_lm():
    lm_lstm = {
        "filename": lbs_dataset.get_arpa_lm_dict()["4gram"],  # TODO: Use LSTM here
        "type": "ARPA",
        "scale": 10,
    }

    train, dev, test = data_setups.get_data_inputs(add_unknown_phoneme_and_mapping=True)
    for corpus in list(dev.values()) + list(test.values()):
        corpus.lm = copy.deepcopy(lm_lstm)
    return train, dev, test
