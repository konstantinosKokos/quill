

def binary_stats(predictions: list[bool], truths: list[bool]) -> tuple[int, int, int, int]:
    tp = sum([x == y for x, y in zip(predictions, truths) if y])
    fn = sum([x != y for x, y in zip(predictions, truths) if y])
    tn = sum([x == y for x, y in zip(predictions, truths) if not y])
    fp = sum([x != y for x, y in zip(predictions, truths) if not y])
    return tp, fn, tn, fp


def macro_binary_stats(tp: int, fn: int, tn: int, fp: int) -> tuple[float, float, float, float]:
    prec = tp / (tp + fp + 1e-08)
    rec = tp / (tp + fn + 1e-08)
    f1 = 2 * prec * rec / (prec + rec + 1e-08)
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    return accuracy, f1, prec, rec

