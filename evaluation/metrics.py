import math
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def compute_TP_TN(label, theta):
    golden_clusters = {}
    output_clusters = {}
    num_docs = len(label)

    # Create golden clusters
    for id, lbl in enumerate(label):
        ids = golden_clusters.get(lbl, set())
        ids.add(id)
        golden_clusters[lbl] = ids

    # Create output clusters
    max_topic = np.argmax(theta, axis=1)
    for doc_id, topic_id in enumerate(max_topic):
        lbl = "Topic_" + str(topic_id)
        ids = output_clusters.get(lbl, set())
        ids.add(doc_id)
        output_clusters[lbl] = ids

    # Compute purity
    count = 0
    for _, docs in output_clusters.items():
        correct_assigned_doc_num = 0
        for _, golden_docs in golden_clusters.items():
            correct_assigned_doc_num = max(correct_assigned_doc_num, len(docs.intersection(golden_docs)))
        count += correct_assigned_doc_num
    purity = count / num_docs

    # Compute NMI
    MI_score = 0.0
    for _, docs in output_clusters.items():
        for _, golden_docs in golden_clusters.items():
            num_correct_assigned_docs = len(docs.intersection(golden_docs))
            if num_correct_assigned_docs == 0.0:
                continue
            MI_score += (num_correct_assigned_docs / num_docs) * math.log(
                (num_correct_assigned_docs * num_docs) / (len(docs) * len(golden_docs)))
    entropy = 0.0
    for _, docs in output_clusters.items():
        entropy += (-1.0 * len(docs) / num_docs) * math.log(1.0 * len(docs) / num_docs)
    for _, docs in golden_clusters.items():
        entropy += (-1.0 * len(docs) / num_docs) * math.log(1.0 * len(docs) / num_docs)
    NMI = 2 * MI_score / entropy

    return purity, NMI


def calculate_purity(label, pred):
    contingency_matrix = metrics.cluster.contingency_matrix(label, pred)
    precision = contingency_matrix / contingency_matrix.sum(axis=0).reshape(1, -1)
    recall = contingency_matrix / contingency_matrix.sum(axis=1).reshape(-1, 1)
    f1 = 2 * (precision * recall) / (precision + recall)
    f1 = np.nan_to_num(f1)
    purity = (
        np.amax(precision, axis=0) * contingency_matrix.sum(axis=0)
    ).sum() / contingency_matrix.sum()
    inverse_purity = (
        np.amax(recall, axis=1) * contingency_matrix.sum(axis=1)
    ).sum() / contingency_matrix.sum()
    harmonic_purity = (
        np.amax(f1, axis=1) * contingency_matrix.sum(axis=1)
    ).sum() / contingency_matrix.sum()

    return (purity, inverse_purity, harmonic_purity)


def rf_cls(train_theta, train_y, test_theta, test_y):
    clf = RandomForestClassifier(n_estimators=10, max_depth=8, random_state=0)

    train_theta = train_theta.astype('float32')
    test_theta = test_theta.astype('float32')
    train_y = train_y.ravel()
    test_y = test_y.ravel()

    clf.fit(train_theta, train_y)
    predict_test = clf.predict(test_theta)
    acc = metrics.accuracy_score(test_y, predict_test)

    return acc


def topic_diversity(topics, topk=10):
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than '+str(topk))
    else:
        unique_words = set()
        for topic in topics:
            unique_words = unique_words.union(set(topic[:topk]))
        puw = len(unique_words) / (topk * len(topics))
        return puw


def read_tc(tc_content, agg_mode='avg_all'):
    result = tc_content.split('\n')[2:-1]
    tcs = []
    for line in result:
        tc = line.split('\t')[1]
        tcs.append(float(tc))

    tcs_np = np.array(tcs)
    if agg_mode == 'avg_all':
        res = np.mean(tcs_np)

    return res