# Evaluate-Spark-MLlib-using-PySpark
A UDF to evaluate Spark-MLlib classification model using PySpark

##### Compute Cohen's kappa coefficient
```
def kappa(tp, tn, fp, fn):
    N = tp+tn+fp+fn
    # Probability observed
    Po = float(tp+tn)/N
    # Probability expected
    Pe = float(((tn+fp)*(tn+fn))+((fn+tp)*(fp+tp)))/(N*N)
    # Cohen's kappa Coefficient
    kappa = float(Po-Pe)/(1-Pe)
    return(kappa)
 ```
 ##### Evaluate the classifier
 ```
 def evaluate(predictions):
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    import math
    
    print "Spam=",result[(result.label == 1)].count(),";No-Spam",result[(result.label == 0)].count()    
    
    eval = BinaryClassificationEvaluator()
    # Area under the ROC curve
    auroc = eval.evaluate(result,{eval.metricName:"areaUnderROC"})
    # Area under the PR curve
    aupr = eval.evaluate(result,{eval.metricName:"areaUnderPR"})
    print "\n The AUROC is %s and the AUPR is %s" %(round(auroc,3), round(aupr,3))
    
    # True Positives
    tp= result[(result.label == 1) & (result.prediction == 1)].count()
    # True Negatives
    tn= result[(result.label == 0) & (result.prediction == 0)].count()
    # False Positives
    fp= result[(result.label == 0) & (result.prediction == 1)].count()
    # False Negatives
    fn= result[(result.label == 1) & (result.prediction == 0)].count()
    print "\n True Positives= %s; True Negatives= %s; False Positives= %s; False Negatives= %s" %(tp, tn, fp, fn)

    # Model Accuracy 
    accuracy = float(tp+tn)/float(tp+tn+fp+fn)
    # Sensitivity/Recall
    recall = float(tp)/(tp+fn)
    # Specificity
    spec = float(tn)/(tn+fp)
    # Precision/PPV
    precision = float(tp)/float(tp+fp)
    # F-measure
    fscore = (2*recall*precision)/(recall+precision)
    # Matthews correlation coefficient
    MCC = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (fp + tn) * (tn + fn))
    # Cohen's kappa coefficient
    cohen_kappa = kappa(tp, tn, fp, fn)
    
    print "\n Accuracy= %s; Sensitivity= %s; Specificity= %s; Precision= %s \n F-measure= %s; 
    Matthews correlation coefficient= %s; Cohen's Kappa coefficient= %s" %
    (round(accuracy*100,2),round(recall*100,2),round(spec*100,2),round(precision*100,2),round(fscore,4),
    round(MCC,4),round(cohen_kappa,4))   
```    
