import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, chi2_contingency, ttest_ind
from statsmodels.stats.multitest import multipletests


SUBTYPE_ORDER = ("MARD", "MOD", "SIDD", "SIRD")
OUTCOMES = {
    "CKD": "CKD_prevalence_analysis",
    "Diabetic retinopathy": "DR_prevalence_analysis",
    "Coronary artery disease": "CAD_prevalence_analysis",
    "Myocardial infarction": "MI_prevalence_analysis",
    "Stroke": "stroke_prevalence_analysis",
}
CLINICAL_COVARIATES = (
    "p_age", "male", "p_bmi", "diabetes_duration",
    "current_smoking", "hypertension", "p_hba1c",
)
MEDICATION_COVARIATES = ("insulin_diabetes", "metformin", "statin")
MODEL_SPECS = {
    "clinical_adjusted": CLINICAL_COVARIATES,
    "fully_adjusted": CLINICAL_COVARIATES + MEDICATION_COVARIATES,
}
SUBTYPE_TERM = "C(subtype, Treatment(reference='MARD'))"


def compare_included_excluded(data, inclusion_column, continuous, categorical):
    rows = []
    included = data[data[inclusion_column] == 1]
    excluded = data[data[inclusion_column] == 0]
    for variable in continuous:
        left, right = included[variable].dropna(), excluded[variable].dropna()
        result = ttest_ind(left, right, equal_var=False)
        rows.append({"variable":variable,"type":"continuous",
            "included_n":len(left),"excluded_n":len(right),
            "included_mean":left.mean(),"included_sd":left.std(),
            "excluded_mean":right.mean(),"excluded_sd":right.std(),
            "p_value":result.pvalue})
    for variable in categorical:
        table = pd.crosstab(data[inclusion_column], data[variable])
        p_value = chi2_contingency(table).pvalue
        rows.append({"variable":variable,"type":"categorical",
            "included_n":included[variable].notna().sum(),
            "excluded_n":excluded[variable].notna().sum(),"p_value":p_value})
    return pd.DataFrame(rows)


def _formula(outcome, covariates, subtype=True):
    terms = list(covariates)
    if subtype:
        terms.insert(0, SUBTYPE_TERM)
    return f"{outcome} ~ " + " + ".join(terms)


def _fit(formula, data):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = smf.logit(formula, data=data).fit(
            method="newton", maxiter=200, disp=False
        )
    return result, " | ".join(str(item.message) for item in captured)


def _coefficient_rows(result, outcome, model):
    confidence = result.conf_int()
    return pd.DataFrame({
        "outcome":outcome,"model":model,"term":result.params.index,
        "coefficient":result.params.to_numpy(),
        "standard_error":result.bse.to_numpy(),
        "odds_ratio":np.exp(result.params.to_numpy()),
        "ci_lower":np.exp(confidence[0].to_numpy()),
        "ci_upper":np.exp(confidence[1].to_numpy()),
        "p_value":result.pvalues.to_numpy(),
    })


def adjusted_complication_models(data, output_dir, outcomes=None,
                                 model_specs=None):
    outcomes = OUTCOMES if outcomes is None else outcomes
    model_specs = MODEL_SPECS if model_specs is None else model_specs
    data = data.copy()
    data["subtype"] = pd.Categorical(
        data["subtype"], categories=SUBTYPE_ORDER, ordered=True
    )
    required = ["subtype", *outcomes.values(), *set(sum(
        (list(value) for value in model_specs.values()), []
    ))]
    missing = sorted(set(required)-set(data.columns))
    if missing:
        raise KeyError(f"Missing model columns: {missing}")
    if not data[list(outcomes.values())].isin([0,1]).all().all():
        raise ValueError("Outcomes must be binary")

    global_rows=[]; coefficient_frames=[]; diagnostic_rows=[]
    for outcome_name,outcome_column in outcomes.items():
        for model_name,covariates in model_specs.items():
            full_formula=_formula(outcome_column,covariates,True)
            reduced_formula=_formula(outcome_column,covariates,False)
            full,full_warnings=_fit(full_formula,data)
            reduced,reduced_warnings=_fit(reduced_formula,data)
            statistic=2*(full.llf-reduced.llf)
            degrees=int(full.df_model-reduced.df_model)
            global_rows.append({"outcome":outcome_name,"model":model_name,
                "likelihood_ratio_chi_square":statistic,
                "degrees_of_freedom":degrees,"p_value":chi2.sf(statistic,degrees),
                "full_formula":full_formula,"reduced_formula":reduced_formula})
            coefficient_frames.append(_coefficient_rows(full,outcome_name,model_name))
            diagnostic_rows.append({"outcome":outcome_name,"model":model_name,
                "converged":bool(full.mle_retvals.get("converged",False)),
                "iterations":full.mle_retvals.get("iterations"),
                "warning":full_warnings,"reduced_warning":reduced_warnings,
                "event_n":int(data[outcome_column].sum()),"participant_n":len(data)})
    global_tests=pd.DataFrame(global_rows)
    for model_name in model_specs:
        mask=global_tests["model"]==model_name
        global_tests.loc[mask,"fdr_p_value"]=multipletests(
            global_tests.loc[mask,"p_value"],method="fdr_bh"
        )[1]
    coefficients=pd.concat(coefficient_frames,ignore_index=True)
    diagnostics=pd.DataFrame(diagnostic_rows)
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    global_tests.to_csv(output_dir/"global_subtype_likelihood_ratio_tests.csv",index=False)
    coefficients.to_csv(output_dir/"all_adjusted_odds_ratios.csv",index=False)
    diagnostics.to_csv(output_dir/"logistic_model_diagnostics.csv",index=False)
    return global_tests,coefficients,diagnostics
