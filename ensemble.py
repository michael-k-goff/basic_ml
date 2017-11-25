# Ensemble model. Take the average of multiple other models

import numpy as np
import sklearn

# Apply the ensemble model to all models in the input, then return the error
def Ensemble(test_X, test_y,models):
    pred_test = sum([m.predict(test_X) for m in models]) / len(models)
    rms_test = sklearn.metrics.mean_squared_error(test_y,pred_test)
    
    return rms_test