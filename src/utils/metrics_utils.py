
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
# metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from cdt.metrics import SHD
import numpy as np
import torch


def get_off_diagonal(A):
    # assumes A.shape: (batch, x, y)
    M = np.invert(np.eye(A.shape[1], dtype=bool))
    return A[:, M]


def adjacency_f1(adj_matrix, predictions):
    # adj_matrix: (b, l, d, d) or (b, d, d)
    # predictions: (b, l, d, d) or (b, d, d)
    D = adj_matrix.shape[-1]
    L = np.tril_indices(D, k=-1)
    U = np.triu_indices(D, k=1)

    adj_lower = adj_matrix[..., L[0], L[1]]
    adj_upper = adj_matrix[..., U[0], U[1]]

    adj_diag = np.diagonal(adj_matrix, axis1=-2, axis2=-1).flatten()
    adj = np.concatenate((adj_diag, np.logical_or(
        adj_lower, adj_upper).flatten().astype(int)))

    pred_diag = np.diagonal(predictions, axis1=-2, axis2=-1).flatten()
    pred_lower = predictions[..., L[0], L[1]]
    pred_upper = predictions[..., U[0], U[1]]
    pred = np.concatenate((pred_diag, np.logical_or(
        pred_lower, pred_upper).flatten().astype(int)))

    return f1_score(adj, pred)


def compute_shd(adj_matrix, preds, aggregated_graph=False):
    assert adj_matrix.shape == preds.shape, f"Dimension of adj_matrix {adj_matrix.shape} should match the predictions {preds.shape}"

    if not aggregated_graph:
        assert len(
            adj_matrix.shape) == 4, "Expects adj_matrix of shape (batch, lag+1, num_nodes, num_nodes)"
        assert len(
            preds.shape) == 4, "Expects preds of shape (batch, lag+1, num_nodes, num_nodes)"
    else:
        assert len(
            adj_matrix.shape) == 3, "Expects adj_matrix of shape (batch, num_nodes, num_nodes)"
        assert len(
            preds.shape) == 3, "Expects preds of shape (batch, num_nodes, num_nodes)"

    shd_score = 0
    if not aggregated_graph:
        shd_inst = 0
        shd_lag = 0
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                adj_sub_matrix = adj_matrix[i, j]
                preds_sub_matrix = preds[i, j]
                shd = SHD(adj_sub_matrix, preds_sub_matrix)
                shd_score += shd
                if j == 0:
                    shd_inst += shd
                else:
                    shd_lag += shd
        return shd_score/adj_matrix.shape[0], shd_inst/adj_matrix.shape[0], shd_lag/adj_matrix.shape[0]
    for i in range(adj_matrix.shape[0]):
        adj_sub_matrix = adj_matrix[i]
        preds_sub_matrix = preds[i]
        shd_score += SHD(adj_sub_matrix, preds_sub_matrix)
        # print(SHD(adj_sub_matrix, preds_sub_matrix))
    return shd_score/adj_matrix.shape[0]


def calculate_expected_shd(scores, adj_matrix, aggregated_graph=False, n_trials=100):
    totals_shd = 0
    for _ in range(n_trials):
        draw = np.random.binomial(1, scores)
        if aggregated_graph:
            shd = compute_shd(adj_matrix, draw,
                              aggregated_graph=aggregated_graph)
        else:
            shd, _, _ = compute_shd(
                adj_matrix, draw, aggregated_graph=aggregated_graph)
        totals_shd += shd

    return totals_shd/n_trials


def evaluate_results(scores,
                     adj_matrix,
                     predictions,
                     aggregated_graph=False,
                     true_cluster_indices=None,
                     pred_cluster_indices=None):

    assert adj_matrix.shape == predictions.shape, "Dimension of adj_matrix should match the predictions"

    abs_scores = np.abs(scores).flatten()
    preds = np.abs(np.round(predictions))
    truth = adj_matrix.flatten()

    # calculate shd

    if aggregated_graph:
        shd_score = compute_shd(adj_matrix, preds, aggregated_graph)
    else:
        shd_score, shd_inst, shd_lag = compute_shd(
            adj_matrix, preds, aggregated_graph)
        f1_inst = f1_score(get_off_diagonal(adj_matrix[:, 0]).flatten(
        ), get_off_diagonal(predictions[:, 0]).flatten())
        f1_lag = f1_score(adj_matrix[:, 1:].flatten(), preds[:, 1:].flatten())

    f1 = f1_score(truth, preds.flatten())
    adj_f1 = adjacency_f1(adj_matrix, predictions)

    preds = preds.flatten()
    zero_edge_accuracy = np.sum(np.logical_and(
        preds == 0, truth == 0))/np.sum(truth == 0)
    one_edge_accuracy = np.sum(np.logical_and(
        preds == 1, truth == 1))/np.sum(truth == 1)

    accuracy = accuracy_score(truth, preds)
    precision = precision_score(truth, preds)
    recall = recall_score(truth, preds)

    try:
        rocauc = roc_auc_score(truth, abs_scores)
    except ValueError:
        rocauc = 0.5

    tnr = zero_edge_accuracy
    tpr = one_edge_accuracy

    print("Accuracy score:", accuracy)
    print("Orientation F1 score:", f1)
    print("Adjacency F1:", adj_f1)
    print("Precision score:", precision)
    print("Recall score:", recall)
    print("ROC AUC score:", rocauc)

    print("Accuracy on '0' edges", tnr)
    print("Accuracy on '1' edges", tpr)
    print("Structural Hamming Distance:", shd_score)
    if not aggregated_graph:
        print("Structural Hamming Distance (inst):", shd_inst)
        print("Structural Hamming Distance (lag):", shd_lag)
        print("Orientation F1 inst", f1_inst)
        print("Orientation F1 lag", f1_lag)
    eshd = calculate_expected_shd(
        np.abs(scores/(np.max(scores)+1e-4)), adj_matrix, aggregated_graph)
    print("Expected SHD:", eshd)
    # also return a dictionary of metrics
    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'adj_f1': adj_f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': rocauc,
        'tnr': tnr,
        'tpr': tpr,
        'shd_overall': shd_score,
        'expected_shd': eshd
    }

    if not aggregated_graph:
        metrics['shd_inst'] = shd_inst
        metrics['shd_lag'] = shd_lag

        metrics['f1_inst'] = f1_inst
        metrics['f1_lag'] = f1_lag

    if pred_cluster_indices is not None and true_cluster_indices is not None:
        metrics['cluster_acc'] = cluster_accuracy(true_idx=true_cluster_indices,
                                                  pred_idx=pred_cluster_indices)
    else:
        _, true_cluster_indices = np.unique(
            adj_matrix, return_inverse=True, axis=0)
        _, pred_cluster_indices = np.unique(
            predictions, return_inverse=True, axis=0)
        metrics['cluster_acc'] = cluster_accuracy(true_idx=true_cluster_indices,
                                                  pred_idx=pred_cluster_indices)

    return metrics


def mape_loss(X_true, x_pred):
    return torch.mean(torch.abs((X_true - x_pred) / X_true))*100


def cluster_accuracy(true_idx, pred_idx):

    assert true_idx.shape == pred_idx.shape, "Shapes must match"
    # first get the confusion matrix
    cm = confusion_matrix(true_idx, pred_idx)

    # next run a linear sum assignment problem to obtain the maximum matching
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)

    # get the maximum matching
    return cm[row_ind, col_ind].sum()/cm.sum()
