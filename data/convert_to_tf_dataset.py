from collections import defaultdict
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def generator_to_tf_dataset_doe4(generator):
    inputs = (defaultdict(list), defaultdict(list))
    for (signalss, offsetss, cutss), (labelss) in tqdm(generator):
        inputs[0]['time_domain_signals'].append(signalss)
        inputs[0]['offset'].append(offsetss)
        inputs[0]['cut'].append(cutss)

        inputs[1]['time_regressor'].append(labelss)

    inputs[0]['time_domain_signals'] = np.concatenate(inputs[0]['time_domain_signals'])
    inputs[0]['offset'] = np.concatenate(inputs[0]['offset'])
    inputs[0]['cut'] = np.concatenate(inputs[0]['cut'])

    inputs[1]['time_regressor'] = np.concatenate(inputs[1]['time_regressor'])

    return tf.data.Dataset.from_tensor_slices(inputs)


def generator_to_tf_dataset_doe4ss(generator):
    inputs = (defaultdict(list), defaultdict(list))
    for (signalss), (machining_errorss, gngss) in tqdm(generator):
        inputs[0]['time_domain_signals'].append(signalss)

        inputs[1]['time_regressor'].append(machining_errorss)
        inputs[1]['time_classifier'].append(gngss)

    inputs[0]['time_domain_signals'] = np.concatenate(inputs[0]['time_domain_signals'])

    inputs[1]['time_regressor'] = np.concatenate(inputs[1]['time_regressor'])
    inputs[1]['time_classifier'] = np.concatenate(inputs[1]['time_classifier'])
    return tf.data.Dataset.from_tensor_slices(inputs)