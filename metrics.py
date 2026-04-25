from collections import defaultdict
import numpy as np

def writer_metrics(y_true, y_pred, writer_ids, min_samples=5):
    correct_by_writer = defaultdict(int)
    total_by_writer = defaultdict(int)

    for yt, yp, wid in zip(y_true, y_pred, writer_ids):
        total_by_writer[wid] += 1
        correct_by_writer[wid] += int(yt == yp)

    acc_by_writer = {
        wid: correct_by_writer[wid] / total
        for wid, total in total_by_writer.items()
        if total >= min_samples
    }

    if not acc_by_writer:
        return {
            "writer_gap": None,
            "min_writer_acc": None,
            "max_writer_acc": None,
            "mean_writer_acc": None,
            "num_writers_used": 0,
        }

    values = list(acc_by_writer.values())

    return {
        "writer_gap": max(values) - min(values),
        "min_writer_acc": min(values),
        "max_writer_acc": max(values),
        "mean_writer_acc": float(np.mean(values)),
        "num_writers_used": len(values),
        "acc_by_writer": acc_by_writer,
    }

# need to make two version, one without per writer evaluation and one with per writer evaluation
def research_score(val_acc, writer_gap, train_acc=None, lambda_gap=0.5, lambda_gen=0.25):
    if writer_gap is None:
        return -999

    gen_gap = 0.0
    if train_acc is not None:
        gen_gap = max(0.0, train_acc - val_acc)

    return val_acc - lambda_gap * writer_gap - lambda_gen * gen_gap