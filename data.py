import numpy as np
import pandas as pd
from sksurv.preprocessing import OneHotEncoder
from sksurv.datasets import load_breast_cancer

def datasets(name="test", path=None):
    if name == "test":
        X, y = load_breast_cancer()
        Xt = OneHotEncoder().fit_transform(X)
        y = y.astype([('event', 'bool'), ('days', 'float64')])
        return Xt, y
    elif name == "ANZSCTS":
        assert path is not None, "Path is not given for ANZSCTS dataset"
        assert "pickle" in path, "Dataset must be in pickle format"
        data = pd.read_pickle(path)
        # data = data[data["DISCHAR_hospital mortality"] != 1]
        data = data[data["censor_days"] >= 0]
        xcols = ['ICU', 'VENT', 'TP_Isolated CABG', 'TP_Other', 'TP_Valve(s) + CABG', 'TP_Valve(s) only', 'Redo', 'AGE', 'Sex', 'Race1', 'DOSA', 'DOD', 'Insur_7.0', 'Insur_DVA', 'Insur_Medicare', 'Insur_Other', 'Insur_Overseas', 'Insur_Private', 'Insur_Self-insured', 'SMO_H', 'SMO_C', 'DB', 'DB_CON_Diet', 'DB_CON_Insulin', 'DB_CON_None', 'DB_CON_Oral', 'HCHOL', 'PRECR', 'DIAL', 'TRANS', 'HG', 'HYT', 'CBVD', 'CBVD_T_CVA', 'CBVD_T_Carotid Test', 'CBVD_T_Coma', 'CBVD_T_RIND/TIA', 'CART', 'PVD', 'LD', 'LD_T_Mild', 'LD_T_Moderate', 'LD_T_Severe', 'IE', 'IMSRX', 'MI', 'MI_T', 'MI_W_1-7 days', 'MI_W_<=6 hours', 'MI_W_>21 days', 'MI_W_>6 hours - <24 hours', 'MI_W_>7 - 21 days', 'ANGRXG', 'ANGRXH', 'ANGRXC', 'CHF', 'CHF_C', 'NYHA_I', 'NYHA_II', 'NYHA_III', 'NYHA_IV', 'SHOCK', 'RESUS', 'ARRT', 'ARRT_A', 'ARRT_AT', 'ARRT_H', 'ARRT_V', 'ARRT_O', 'PACE', 'MEDIN', 'MEDNI', 'MEDAC', 'MEDST', 'MED_ASP_9.0', 'MED_ASP_No', 'MED_ASP_Yes', 'MED_ASP_W', 'MED_CLOP', 'MED_TICA', 'MED_AGG', 'MED_ABCI', 'MED_OTH', 'POP', 'PTCA', 'PTCA_ADM', 'HTM', 'WKG', 'CATH', 'EF', 'EF_EST_ Mod 30-45%', 'EF_EST_ Severe <30%', 'EF_EST_Mild 46-60%', 'EF_EST_Normal >60%', 'LMD', 'DISVES_none', 'DISVES_one', 'DISVES_three', 'DISVES_two', 'BMI', 'BSA', 'eGFR_Corrected', 'STAT_Elective', 'STAT_Emergency', 'STAT_Salvage', 'STAT_Urgent', 'DTCATH', 'CCAB', 'CVLV', 'COTH', 'CT', 'OTHCON', 'MIN', 'ROBOT', 'CPB', 'CPLEG', 'CCT', 'PERF', 'MINHT', 'IABP', 'ECMO', 'VAD', 'ANTIFIB_9.0', 'ANTIFIB_no', 'ANTIFIB_yes', 'ANTIFIB_T_4.0', 'ANTIFIB_T_other', 'ANTIFIB_T_tranexamic acid', 'ANTIFIB_T_trasylol', 'RBC', 'RBCUnit', 'NRBC', 'PlateUnit', 'NovoUnit', 'CryoUnit', 'FFPUnit', 'REICU', 'REINT', 'DRAIN_4', 'RTT', 'NRF', 'POSTCR', 'POMI', 'POCS', 'POSTHG', 'CIUSE', 'IULOWOUT', 'IULowSVR', 'NARRT', 'HB', 'BA', 'CA', 'AFIB', 'NARRTV', 'CVA_P', 'CVA_T', 'COMA', 'PUEMB', 'PUPNU', 'INFDS', 'SWI', 'DOWI', 'INFTH', 'INFSP', 'NAODS', 'LISCH_2.0', 'LISCH_3.0', 'LISCH_No', 'LISCH_Yes', 'ACOAG', 'GIT', 'MSF', 'DISCHAR_6.0', 'DISCHAR_home', 'DISCHAR_hospital in the home', 'DISCHAR_local/referring hospital', 'DISCHAR_rehabilitation unit/hospital']
        ycols = ['Final_Deathdate', 'censor_days']
        y = np.array([(False, x[1]) if pd.isna(x[0]) else (True, x[0]) for x in data[ycols].values], dtype=[('event', 'bool'), ('days', 'float64')])
        X = data[xcols]
        return X, y
    else:
        raise NotImplementedError("Dataset is not implemented yet")
