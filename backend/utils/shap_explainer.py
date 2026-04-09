import shap
import numpy as np

explainer = None

def get_shap_values(features, model):
    global explainer

    if explainer is None:
        explainer = shap.Explainer(model)

    shap_values = explainer(np.array([features]))

    return list(shap_values.values[0])