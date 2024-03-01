from ingester3.extensions import *
import pandas as pd
import numpy as np

OUT_PATH='.'

def escalation(x):
    """
    Gives labels for escalation, from -6 to 6:
    label 0 : no change in conflict dynamics, for anything between a -25% to +25% change
    label +/- 1 : -50% to +50% change
    label +/- 2 : -75% to +75% change
    label +/- 3 : -100% to +100% change
    label +/- 4 : -500% to +500% change
    label +/- 5 : true onsets and terminations
    
    :param x: 
    :return: 
    """
    if  -0.25 <= x <= 0.25: return 0 
    if  -0.5 <= x <= 0.5: return 1 * np.sign(x)
    if  -0.75 <= x <= 0.75: return 2 * np.sign(x)
    if  -1 <= x <= 1: return 3 * np.sign(x)
    if  -5 <= x <= 5: return 4 * np.sign(x)
    return 5 * np.sign(x)

try:
    # Fetch it from the Dropbox link
    ged = pd.read_csv('https://www.dropbox.com/scl/fi/y6t5fufiswtt1fl8e8xtn/GEDEvent_v23_1.csv?rlkey=3b6ldvu6xfld2sgb8rn23lh33&dl=1', low_memory=False)
except Exception as e:
    print("Not able to download from Dropbox, trying local file...")
    ged = pd.read_csv('GedEvent_v23_1.csv', low_memory=False)


ged['date_end'] = pd.to_datetime(ged.date_end)
ged['date_start'] = pd.to_datetime(ged.date_start)
ged = pd.DataFrame.m.from_datetime(ged, datetime_col='date_start')
ged = ged.rename(columns={'month_id': 'month_id_start'})
ged = pd.DataFrame.m.from_datetime(ged, datetime_col='date_end')

dyads = ged[['dyad_name', 'dyad_new_id']].drop_duplicates()
dyads = dyads.set_index('dyad_new_id')

# Aggregate the GED data to the monthly level using the best and high estimates

ged = ged.query('month_id==month_id_start')[['id', 'month_id', 'dyad_new_id', 'best', 'high']]
ged = ged.groupby(['month_id', 'dyad_new_id']).sum().reset_index()

# Create a panel with all dyads and all months and merge the GED data in 

ged_panel = pd.DataFrame(
    [(i, j) for i in list(ged.dyad_new_id.unique()) for j in range(ged.month_id.min(), ged.month_id.max() + 1)],
    columns=['dyad_new_id', 'month_id']).merge(ged, on=['dyad_new_id', 'month_id'], how='left')

ged_panel.sort_values(['dyad_new_id', 'month_id'], inplace=True)
ged_panel.fillna(0, inplace=True)
del ged_panel['id']
ged_panel['best'] = ged_panel['best'].astype(int)
ged_panel['high'] = ged_panel['high'].astype(int)

# Compute rolling averages for 1, 3, 6, 12 months excluding the current month

ged1 =ged_panel.groupby('dyad_new_id').rolling(window=1, on='month_id', closed='left').mean().reset_index()[['dyad_new_id','month_id','best','high']]
ged1.columns=['dyad_new_id','month_id','best_1','high_1']

ged3 = ged_panel.groupby('dyad_new_id').rolling(window=3, on='month_id', closed='left').mean().reset_index()[['dyad_new_id','month_id','best','high']]
ged3.columns=['dyad_new_id','month_id','best_3','high_3']

ged6 = ged_panel.groupby('dyad_new_id').rolling(window=6, on='month_id', closed='left').mean().reset_index()[['dyad_new_id','month_id','best','high']]
ged6.columns=['dyad_new_id','month_id','best_6','high_6']

ged12 = ged_panel.groupby('dyad_new_id').rolling(window=12, on='month_id', closed='left').mean().reset_index()[['dyad_new_id','month_id','best','high']]
ged12.columns=['dyad_new_id','month_id','best_12','high_12']

ged_panel_full = ged_panel.merge(ged1, on=['dyad_new_id','month_id'], how='left').merge(ged3, on=['dyad_new_id','month_id'], how='left').merge(ged6, on=['dyad_new_id','month_id'], how='left').merge(ged12, on=['dyad_new_id','month_id'], how='left')

# Compute deltas vs the different lags
for i in [1,3,6,12]:
    ged_panel_full[f'delta_{i}'] = (ged_panel_full.best - ged_panel_full[f'best_{i}']) / (ged_panel_full[f'best_{i}'] + 1e-15)
    ged_panel_full[f'delta_high_{i}'] = (ged_panel_full.high - ged_panel_full[f'high_{i}']) / (ged_panel_full[f'high_{i}'] + 1e-15)
    ged_panel_full[f'escalation_{i}'] = ged_panel_full[f'delta_{i}'].apply(escalation)
    ged_panel_full[f'escalation_high_{i}'] = ged_panel_full[f'delta_high_{i}'].apply(escalation)

# Some cleanup for niceness:    
ged_panel_full = ged_panel_full[ged_panel_full.month_id>=ViewsMonth.from_year_month(2010,1).id]
ged_panel_full['dyad_name'] = ged_panel_full.dyad_new_id.apply(lambda i: dyads.loc[i].dyad_name)
ged_panel_full.rename(columns={'dyad_new_id':'dyad_id'}, inplace=True)

ged_panel_full.to_parquet(OUT_PATH.strip().rstrip('/')+'ged_escalation.parquet')
