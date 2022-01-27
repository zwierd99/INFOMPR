import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


class StatisticalComparison:
    def __init__(self, y_true, y_pred_1, y_pred_2):
        self.y_true = y_true
        self.y_pred_1 = y_pred_1
        self.y_pred_2 = y_pred_2
        self.mcnemar()

    def mcnemar(self):
        binarized_answer_1 = [1 if self.y_pred_1[x] == self.y_true[x] else 0 for x in range(len(self.y_true))]
        binarized_answer_2 = [1 if self.y_pred_2[x] == self.y_true[x] else 0 for x in range(len(self.y_true))]

        df = pd.DataFrame(list(zip(binarized_answer_1, binarized_answer_2)), columns=['clf1', 'clf2'])
        contingency_table = pd.crosstab(df["clf1"], df["clf2"])
        stat_res = mcnemar(contingency_table)
        print(stat_res)
