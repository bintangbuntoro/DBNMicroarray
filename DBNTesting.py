import numpy as np


def SigmoidFunction(x):
    return 1 / (1.0 + np.exp(-x))


def _transform_labels_to_network_format(labels, num_classes):
    new_labels = np.zeros([len(labels), num_classes])
    label_to_idx_map, idx_to_label_map = dict(), dict()
    idx = 0
    for i, label in enumerate(labels):
        if label not in label_to_idx_map:
            label_to_idx_map[label] = idx
            idx_to_label_map[idx] = label
            idx += 1
        new_labels[i][label_to_idx_map[label]] = 1
    return new_labels, idx_to_label_map


def _transform_network_format_to_labels(indexes, y):
    new_labels, idx_to_label_map = _transform_labels_to_network_format(y, 2)
    return map(lambda idx: idx_to_label_map[idx], indexes)


def _compute_output_units_matrix(matrix_visible_units, W, b):
    matrix_scores = np.transpose(np.dot(W, np.transpose(matrix_visible_units)) + b[:, np.newaxis])
    exp_scores = np.exp(matrix_scores)
    return exp_scores / np.expand_dims(np.sum(exp_scores, axis=1), 1)


def _compute_hidden_units_matrix(matrix_visible_units, W, c):
    return np.transpose(SigmoidFunction(np.dot(W, np.transpose(matrix_visible_units)) + c[:, np.newaxis]))


def _compute_hidden_units(vector_visible_units, W, c):
    v = np.expand_dims(vector_visible_units, 0)
    return np.squeeze(_compute_hidden_units_matrix(v, W, c))


def calcultetransform(X, W, c):
    if len(X.shape) == 1:  # It is a single sample
        return _compute_hidden_units(X, W, c)
    transformed_data = _compute_hidden_units_matrix(X, W, c)
    return transformed_data


def transform(X, listWeight, listBias):
    input_data = X
    for rbm in range(0, len(listWeight) - 1):
        input_data = calcultetransform(input_data, listWeight[rbm], listBias[rbm])
    return input_data


def predictData(X, listWeight, listBias):
    if len(X.shape) == 1:
        X = np.expand_dims(X, 0)
    transformed_data = transform(X, listWeight, listBias)
    predicted_data = _compute_output_units_matrix(transformed_data, listWeight[-1], listBias[-1])
    return predicted_data


def predict(X, y, listWeight, listBias):
    probs = predictData(X, listWeight, listBias)
    indexes = np.argmax(probs, axis=1)
    pred = _transform_network_format_to_labels(indexes, y)
    return pred
