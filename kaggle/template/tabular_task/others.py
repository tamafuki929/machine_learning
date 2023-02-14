## opt prediction threshold in terms of auc
def opt_threshold(y_pred_proba, y):
    thresholds = np.arange(0, 1, 0.01)
    maxv = 0
    ans = -1
    for threshold in thresholds:
        y_pred = np.where(y_pred_proba < threshold, 0, 1)
        if maxv < roc_auc_score(y, y_pred):
            ans  = threshold
            maxv = roc_auc_score(y, y_pred)
    return ans, maxv, accuracy_score(y, np.where(y_pred_proba < ans, 0, 1))