#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

data = pd.read_pickle("./temp/anzscts.pickle")

data = data.loc[data.censor_days > 0]
print(data.shape)
print(list(data))

ycols = ['Final_Deathdate', 'censor_days']
y = np.array([(0, x[1]) if pd.isna(x[0]) else (1, x[0]) for x in data[ycols].values], dtype=[('event', 'float64'), ('days', 'float64')])
T = y["days"]
E = y["event"]
sex = data.Sex.values
race = data.Race1.values
cabg = data["TP_Isolated CABG"].values
valve = data["TP_Valve(s) only"].values
both = data["TP_Valve(s) + CABG"].values
other = data["TP_Other"].values

idx = T > 0
T = T[idx]
E = E[idx]
sex = sex[idx]
race = race[idx]
cabg = cabg[idx]
valve = valve[idx]
both = both[idx]
other = other[idx]

fig, axes = plt.subplots(1, 3)

kmf = KaplanMeierFitter()
kmf.fit(T[sex==1.0], event_observed=E[sex==1.0], label="Female")
kmf.plot_survival_function(ax=axes[0])

kmf = KaplanMeierFitter()
kmf.fit(T[sex==0.0], event_observed=E[sex==0.0], label="Male")
kmf.plot_survival_function(ax=axes[0])


kmf = KaplanMeierFitter()
kmf.fit(T[race==1.0], event_observed=E[race==1.0], label="Indigenous")
kmf.plot_survival_function(ax=axes[1])

kmf = KaplanMeierFitter()
kmf.fit(T[race==0.0], event_observed=E[race==0.0], label="Non-indigenous")
kmf.plot_survival_function(ax=axes[1])


kmf = KaplanMeierFitter()
kmf.fit(T[cabg==1.0], event_observed=E[cabg==1.0], label="Isolated CABG")
kmf.plot_survival_function(ax=axes[2])

kmf = KaplanMeierFitter()
kmf.fit(T[valve==1.0], event_observed=E[valve==1.0], label="Valve(s) only")
kmf.plot_survival_function(ax=axes[2])

kmf = KaplanMeierFitter()
kmf.fit(T[both==1.0], event_observed=E[both==1.0], label="CABG + Valve(s)")
kmf.plot_survival_function(ax=axes[2])

kmf = KaplanMeierFitter()
kmf.fit(T[other==1.0], event_observed=E[other==1.0], label="Other")
kmf.plot_survival_function(ax=axes[2])

axes[0].set_xlabel("")
axes[0].set_title("(a)")
axes[1].set_xlabel("")
axes[1].set_title("(b)")
axes[2].set_xlabel("")
axes[2].set_title("(c)")

# add a big axis, hide frame
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

plt.xlabel("Days from Surgery")
plt.ylabel("Probability of Survival")

plt.show()
