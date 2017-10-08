from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from adspy_shared_utilities import plot_class_region_for_classifier
import matplotlib as plt
import numpy as np
from sklearn.svm import SVC

dataset = load_digits()

X, y = dataset.data, dataset.target == 1
# train_test_split()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

jitter_delta = 0.25
X_twover_train = X_train[:, [20, 59]] + np.random.rand(X_train.shape[0], 2) - jitter_delta
X_twover_test = X_test[:, [20, 59]] + np.random.rand(X_test.shape[0], 2) - jitter_delta

clf = SVC(kernel='linear').fit(X_twover_train, y_train)
gride_values = {'class_weight':
                    ['balanced', {1: 2}, {1: 3}, {1: 4}, {1: 5}, {1: 10}, {1: 20}, {1: 50}]}
plt.figure()

for i, eval_metrics in enumerate(('precision', 'recall', 'f1', 'roc_auc')):
    gride_clf_custom = GridSearchCV(clf, param_grid=gride_values, scoring=eval_metrics)
    gride_clf_custom.fit(X_twover_train, y_train)
    print('Grid best parameter (max. {0}): {1}'.format(eval_metrics, gride_clf_custom.best_params_))
    print('Grid best score (max. {0}): {1}'.format(eval_metrics, gride_clf_custom.best_score_))

    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plot_class_region_for_classifier(gride_clf_custom, X_twover_test, y_test)
    plt.title(eval_metrics + '-oriented SVC')

    plt.show()
