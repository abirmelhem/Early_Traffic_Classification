"""Saved-prediction analysis tests, with synthetic predictions and no models."""
import unittest
import pandas as pd

from src.analysis.ml_results import class_changes, role_summary, verify_metrics
from src.ml.features import BUDGETS, CLASSES
from src.ml.metrics import calculate_metrics


class ResultsAnalysisTest(unittest.TestCase):
    def fixtures(self):
        predictions=[];pooled=[];folds=[]
        for n in BUDGETS:
            for fold in range(1,5):
                for cls in CLASSES:
                    predictions.append(dict(flow_id=f"{fold}-{cls}",n_packets=n,fold=fold,y_true=cls,y_pred=cls))
                _,metrics,_=calculate_metrics(CLASSES,CLASSES)
                folds.append(metrics.assign(n_packets=n,fold=fold))
            _,metrics,_=calculate_metrics(list(CLASSES)*4,list(CLASSES)*4)
            pooled.append(metrics.assign(n_packets=n))
        return pd.DataFrame(predictions),pd.concat(pooled,ignore_index=True),pd.concat(folds,ignore_index=True)

    def test_saved_metrics_match_predictions(self):
        pred,pooled,folds=self.fixtures()
        verify_metrics(pred,pooled,folds)

    def test_inconsistent_saved_metrics_rejected(self):
        pred,pooled,folds=self.fixtures()
        pooled.loc[0,"f1"]=.2
        with self.assertRaisesRegex(ValueError,"disagrees"):
            verify_metrics(pred,pooled,folds)

    def test_all_tied_observed_budgets_retained(self):
        _,pooled,_=self.fixtures()
        changes=class_changes(pooled.assign(model="synthetic"))
        self.assertTrue(changes.best_observed_n.eq("3;5;10;20").all())
        self.assertTrue(changes.delta_5_to_20.eq(0).all())

    def test_role_summary_uses_original_training_roles(self):
        _,_,folds=self.fixtures()
        folds=folds.assign(model="synthetic")
        roles=folds[["model","n_packets","fold","candidate_class"]].copy()
        roles["role"]=roles.candidate_class.map({"CHAT":"minority","FT":"minority","MAIL":"majority","VOIP":"majority"})
        _,per_fold,summary=role_summary(folds,roles)
        self.assertEqual(len(per_fold),32)
        self.assertTrue(per_fold.classes.eq(2).all())
        self.assertTrue(summary.fold_mean.eq(1).all())
        self.assertTrue(summary.fold_std.eq(0).all())


if __name__=="__main__":unittest.main()
