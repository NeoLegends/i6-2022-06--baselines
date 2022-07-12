import copy

import i6_experiments.common.datasets.librispeech as lbs_dataset
import i6_experiments.users.raissi.experiments.librispeech.data_preparation.other_960h.pipeline_base_args as data_setups

def get_data_4gram_lm():
    return data_setups.get_data_inputs()

def get_data_lstm_lm():
    lm_lstm = {
        "filename": lbs_dataset.get_arpa_lm_dict()["4gram"], # TODO: Use LSTM here
        "type": "ARPA",
        "scale": 10,
    }

    train, dev, test = data_setups.get_data_inputs()
    for corpus in list(dev.values()) + list(test.values()):
        corpus.lm = copy.deepcopy(lm_lstm)
    return train, dev, test
