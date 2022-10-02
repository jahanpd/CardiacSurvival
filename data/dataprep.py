import pandas as pd
import numpy as np
import json
from scipy import stats

path = "/Volumes/shared/R-MNHS-SPHPM-DATA/ANZSCTS/Jahan-Penny-Dimiri/update_data_2020/With NDI/ANZSCTS Research Datacut 2019 (Penny-Dimri)_NDI.sav"

def get_anzscts():
# 1. import data

# 2. import subset json dict

# 3. for each feature determine if categorical or numerical
# 3a. if numerical then add to dataset if less than 10% missing values
# 3b. if categorical then determine if binary
# 3c. if binary then add if less than 10% missing values
# 3d. if categorical and less than 10% missingness then dummify and add to dataset
# 3e. record dummified variable names in the json dict

# 4. ensure date of discharge is less than date of death

# 5. convert cath_w to time in days wrt DOP and delete DOP
# 5. convert DOD and Final_Deathdate to time in days and delete them bothnumerical

    # special case variables
    nodummy = ["DEATH_CODE","MORT_R",]

    print("getting data")
    try:
        data = pd.read_pickle('../temp/full.pickle')
    except:
        data = pd.read_spss(path)
        data.to_pickle('../temp/full.pickle')

    print("data retrieved")
    # rescale constant for datetime as spss and datetime differ
    origin = 12219379200
    # censor date
    censor_date = pd.to_datetime("2019-8-1")
    # rescale DOP
    data["DOP"]=pd.to_datetime(data.DOP-12219379200, unit="s")
    # replace -1 to nan as no columns use a genuine -1 value
    data.replace(-1, np.nan, inplace=True)
    data.replace("-1", np.nan, inplace=True)
    data.replace("unknown", np.nan, inplace=True)
    data.replace(" Not Specified", np.nan, inplace=True)
    
    with open("subset.json") as file:
        data_dict = json.load(file)

    test1 = [x for x in data_dict.keys() if x not in list(data)]
    assert len(test1) == 0, "Features in data_dict not in imported data"
    subset = data[list(data_dict.keys())]

    dd_new = {}
    ds_new = {}
    for k,v in data_dict.items():
        if v["type"] == "numerical":
            print(k, " numerical")
            series = data[k]
            nan_frac= series.isna().sum()/len(series)
            if nan_frac > 0.9:
                continue
            else:
                ds_new[k] = series.values
                # add summary statistics
                summary = {}
                summary["mean"] = series.mean()
                summary["std"] = series.std()
                mort = series[~pd.isna(data.Final_Deathdate)]
                alive = series[pd.isna(data.Final_Deathdate)]
                summary["mean_mort"] = mort.mean()
                summary["std_mort"] = mort.std()
                summary["mean_alive"] = alive.mean()
                summary["std_alive"] = alive.std()
                _, summary["pvalue"] = stats.ttest_ind(mort.dropna().values, alive.dropna().values)
                v["summary"] = summary
                dd_new[k] = v
        elif v["type"] == "categorical":
            df = data[[k]]
            nan_frac= df[k].isna().sum()/len(df)
            _, cats = pd.factorize(df[k])
            values = list(cats.values)
            if nan_frac > 0.9:
                print("Excess missing for {}".format(k))
                continue
            else:
                if len(values) == 2:
                    print(k, " binary")
                    v["values"] = values
                    # add summary statistics
                    summary = {}
                    summary["counts"] = df[k].value_counts().to_dict()
                    mort = df[k][~pd.isna(data.Final_Deathdate)]
                    alive = df[k][pd.isna(data.Final_Deathdate)]
                    summary["counts_mort"] = mort.value_counts().to_dict()
                    summary["counts_alive"] = alive.value_counts().to_dict()
                    if summary["counts_alive"].keys() != summary["counts_mort"].keys():
                        diff = set(summary["counts_alive"].keys()).symmetric_difference(set(summary["counts_mort"].keys()))
                        if len(summary["counts_alive"].keys()) > len(summary["counts_mort"].keys()):
                            for d in diff:
                                summary["counts_mort"][d]=0.0
                        else:
                            for d in diff:
                                summary["counts_alive"][d]=0.0
                    table = pd.crosstab(df[k], ~pd.isna(data.Final_Deathdate))
                    summary["pvalue"] = stats.chi2_contingency(table.values)[1]
                    v["summary"] = summary
                    dd_new[k] = v
                    ds_new[k] = [np.nan if pd.isna(x) else values.index(x) for x in df[k].values]
                elif len(values) > 2:
                    print(k, " categorical")
                    v["values"] = values
                    # add summary statistics
                    summary = {}
                    summary["counts"] = df[k].value_counts().to_dict()
                    mort = df[k][~pd.isna(data.Final_Deathdate)]
                    alive = df[k][pd.isna(data.Final_Deathdate)]
                    summary["counts_mort"] = mort.value_counts().to_dict()
                    summary["counts_alive"] = alive.value_counts().to_dict()
                    if summary["counts_alive"].keys() != summary["counts_mort"].keys():
                        diff = set(summary["counts_alive"].keys()).symmetric_difference(set(summary["counts_mort"].keys()))
                        if len(summary["counts_alive"].keys()) > len(summary["counts_mort"].keys()):
                            for d in diff:
                                summary["counts_mort"][d]=0.0
                        else:
                            for d in diff:
                                summary["counts_alive"][d]=0.0
                    table = pd.crosstab(df[k], ~pd.isna(data.Final_Deathdate))
                    summary["pvalue"] = stats.chi2_contingency(table.values)[1]
                    v["summary"] = summary
                    if k not in nodummy:
                        dummified = pd.get_dummies(df, dummy_na=True)
                        nans = dummified[list(dummified)[-1]]
                        v["columns"] = list(dummified)[:-1]
                        dd_new[k] = v
                        for col in v["columns"]:
                            ds_new[col] = [x if y == 0 else np.nan for x,y in zip(dummified[col],nans)]
                    else:
                        dd_new[k] = v
                        ds_new[k] = df[k]
                else:
                    raise ValueError("Length of categories < 2 at {} for {}".format(len(values), k))
        elif v["type"] == "datetime":
            if k == "DOP":
                dd_new[k] = v
                ds_new[k] = data.DOP
            else:
                v["type"] = "numerical"
                dd_new[k] = v
                ds_new[k] = (pd.to_datetime(data[k]-origin, unit="s") - data.DOP).dt.days
        else:
            raise ValueError("Numerical or Categorical labelling error in datadict for {}".format(k))

    # add censoring days
    ds_new["censor_days"] = (censor_date - data.DOP).dt.days

    with open('data_dictionary.json', 'w') as outfile:
        json.dump(dd_new, outfile)
    outfile.close()

    prepped = pd.DataFrame(ds_new)
    prepped.to_pickle('../temp/anzscts.pickle')
    print(prepped.head())


get_anzscts()
