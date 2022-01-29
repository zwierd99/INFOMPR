# 3rd party Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import seaborn as sn
# TensorFlow Dependencies
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout,
)

from statsmodels.stats.contingency_tables import mcnemar
from src.machinelearning import learning_rate, batch_size



def nice_confusion_matrix(cm):
    df_cm = pd.DataFrame(cm)
    hm = sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, cmap="Blues", fmt="g",
                    norm=LogNorm(), cbar=False)
    hm.xaxis.tick_top()
    hm.xaxis.set_label_position('top')
    plt.xticks(np.arange(0, 1, step=1/10))
    plt.xticks(np.arange(10)+0.5, ["Blues", "Classical", "Country", "Disco", "Hiphop", "Jazz", "Metal", "Pop", "Reggae", "Rock"],
               rotation=45)
    plt.yticks(np.arange(10)+0.5, ["Blues", "Classical", "Country", "Disco", "Hiphop", "Jazz", "Metal", "Pop", "Reggae", "Rock"],
               rotation=0
    )
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.show()


def plot_f1_scores(f1_scores):
    fig = plt.figure(figsize = (10, 5))
    labels = ["Blues", "Classical", "Country", "Disco", "Hiphop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]
    plt.bar(labels, f1_scores, color=(47/255, 101/255, 189/255),
            width=0.4)
    plt.ylabel("F1-score")
    plt.xlabel("Genres")
    plt.title("F1-score per genre")

    # plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.show()


def evaluate_model(model, X_test, y_test, checkpoint_filepath=None):
    X_test = np.array(X_test.tolist())

    if checkpoint_filepath:
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.load_weights(checkpoint_filepath)

    _, test_acc = model.evaluate(X_test, y_test, verbose=2, batch_size=batch_size, )
    y_prob = model.predict(X_test)
    y_pred_mel = y_prob.argmax(axis=-1)
    print(classification_report(y_test,y_pred_mel))
    plot_f1_scores(f1_score(y_test, y_pred_mel, average=None))
    nice_confusion_matrix(confusion_matrix(y_test, y_pred_mel))

    print(f"Test Accuracy: {test_acc}")

def mcnemar_test(y_true, y_pred_1, y_pred_2):
    binarized_answer_1 = [1 if y_pred_1[x] == y_true[x] else 0 for x in range(len(y_true))]
    binarized_answer_2 = [1 if y_pred_2[x] == y_true[x] else 0 for x in range(len(y_true))]

    df = pd.DataFrame(list(zip(binarized_answer_1, binarized_answer_2)), columns=['clf1', 'clf2'])
    contingency_table = pd.crosstab(df["clf1"], df["clf2"])
    stat_res = mcnemar(contingency_table)
    print(stat_res)
