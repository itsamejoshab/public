import random
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from mockseries.trend import LinearTrend, Switch
from mockseries.noise import RedNoise
from mockseries.utils import datetime_range
from mockseries.seasonality import SinusoidalSeasonality, YearlySeasonality
from mockseries.transition  import LambdaTransition 
from mockseries.utils.timedeltas import JANUARY, MARCH, JUNE, JULY, AUGUST, SEPTEMBER, OCTOBER, DECEMBER

# random seed
RNG = np.random.default_rng(1000000001)

###########################################
# CONFIGS FOR FAKE DATA
# 
# to make categorical variables
def make_cats(list_vals, list_dist, size):
    return RNG.choice(list_vals, size, p=list_dist)

# to make numeric variables with gamma distributions
def make_gamma(shape, scale, shift, constant, size, round_flag):
    if round_flag == True:
        return np.round(RNG.gamma(shape, scale, size)+shift, decimals=0).astype('int64') * constant
    else: 
        return RNG.gamma(shape, scale, size) + shift * constant
    
# to make numeric variables with poisson distributions
def make_poisson(avg, size):
    return RNG.poisson(avg, size)

###########################################

# seed a base dataset with only 100 rows
# generate the platforms/tactics to a distibution
# will expand later for multiple dates
N_ROWS = 100

config_platforms = {"platform": (make_cats, { "list_vals": ['Google Ads', 'LinkedIn', 'Meta', 'TikTok','Instagram','Twitch.tv','Bing']
                               ,"list_dist": [0.33, 0.29, 0.21, 0.09, 0.05, 0.01, 0.02]
                               ,"size": N_ROWS})}

# execute the config for fake data
df_platforms = pd.DataFrame({k:v[0](**v[1]) for (k,v) in config_platforms.items()})

# manually map each platform to a tactic
df_platforms['tactic'] = np.where(df_platforms['platform'].isin(['LinkedIn','Meta','TikTok','Instagram','Twitch.tv']), 'Paid Social',
                                np.where(df_platforms['platform'].isin(['Google Ads','Bing']), 'Paid Search',
                                         'Other'))

# base dataset of unique dates
df_dates = pd.DataFrame(pd.date_range(start='2021-01-01', end='2023-08-31'), columns=['date'])

# platforms and unique dates - all combos
df_base = df_dates.merge(df_platforms, how='cross')

# reset the N_ROWS
N_ROWS=len(df_base)

# configs for new columns
config = {
    "clicks_pct" : (make_gamma, {"shape": 3, "scale": 2.5, "shift": 1, "constant": 1, "round_flag": True, "size": N_ROWS}),
    "purchases_pct": (make_poisson, {"avg":50, "size":N_ROWS}),
    "impressions": (make_gamma, {"shape": 0.25, "scale": 350, "shift": 100, "constant": 1, "round_flag": True, "size": N_ROWS}),
    "rev_per" : (make_gamma, {"shape": 3, "scale": 1, "shift": 26, "constant": 1, "round_flag": False, "size": N_ROWS}),  
    "cost_per" : (make_gamma, {"shape": 3, "scale": 1, "shift": 25, "constant": 1, "round_flag": False, "size": N_ROWS}),  
}

# execute the config for fake data
df = pd.DataFrame({k:v[0](**v[1]) for (k,v) in config.items()})

# merge the 2 fake data dataframes
merged = pd.concat([df_base,df], axis=1)

# add a campaign name 
# the complexity here is just to make it look messy and semi realistic
merged['campaign_name'] = np.where(merged['platform']=='TikTok', 'TikTok Monthly ' + merged['date'].dt.strftime('%B') + ' ' + merged['tactic'],
                                  np.where(merged['platform']=='LinkedIn', merged['tactic'] + ' campaign for ' + merged['date'].dt.strftime('%B'),
                                           np.where(merged['platform']=='Google Ads', merged['date'].dt.strftime('%Y') + ' Google ' + merged['tactic'] + ' campaign',
                                          merged['platform'] + ' campaign (' + merged['tactic'] + ')' )))

# convert these to percentages
for col in ['clicks_pct','purchases_pct']:
    merged[col] = merged[col]/100
    
# TODO: find a better way to do this
# reduce some of the crazy outliers from the gamma distribution
merged['impressions'] = np.where(merged['impressions']>=10000, 
                                 merged['impressions']*0.9,
                                 merged['impressions']*1.0).round()

####################################
# HELPER FUNCTION FOR ADDING PATTERNS
def add_helper_series(df, result_col_name):
    ''' Takes a dataframe and a name for the result column, which is a combination of all the patterns that are configured in the function '''
    func_end_result=[]
    
    # loop over each combination of platform and tactic
    # so they have different patterns
    for plat in df['platform'].unique():
        temp = df[df['platform']==plat]
        for tac in temp['tactic'].unique():
            
            # original data that is filtered
            original = temp[temp['tactic']==tac]
            
            # unique dates for which to generate the pattern weights
            dts = original['date'].unique()

            # ** SWITCH PATTERN 1 **
            # the transition is meant to make it look realistic
            acceleration = lambda x : 1 - (1-x)**4
            deceleration = lambda x: 3.11111 * x**3 - 7.86667 * x**2 + 5.75556 * x

            realistic_transition = LambdaTransition(
                transition_window=timedelta(days=random.randint(90, 600)),
                transition_function=acceleration,
                stop_window=timedelta(days=random.randint(120, 365)),
                stop_function=deceleration
            )
            start = random.randint(90, 700)
            stop = start + random.randint(30, 200)
            realistic_speed = Switch(
                start_time= pd.to_datetime(dts[start]) + timedelta(days=random.randint(30, 120)),
                base_value=0.7, 
                switch_value=1.1,
                stop_time=pd.to_datetime(dts[stop]) + timedelta(days=random.randint(30, 365)),
                transition=realistic_transition
                )
            
            # ** SWITCH PATTERN 2 **
            acceleration = lambda x : 1 - (1-x)**4
            deceleration = lambda x: 3.11111 * x**3 - 7.86667 * x**2 + 5.75556 * x

            realistic_transition = LambdaTransition(
                transition_window=timedelta(days=random.randint(90, 600)),
                transition_function=acceleration,
                stop_window=timedelta(days=random.randint(120, 365)),
                stop_function=deceleration
            )
            start = random.randint(90, 700)
            stop = start + random.randint(30, 200)
            realistic_speed2 = Switch(
                start_time= pd.to_datetime(dts[start]) + timedelta(days=random.randint(30, 120)),
                base_value=0.9, 
                switch_value=1.4,
                stop_time=pd.to_datetime(dts[stop]) + timedelta(days=random.randint(30, 365)),
                transition=realistic_transition
                )
            
            # ** MONTHLY SEASONAL PATTERN **
            constraints = {
                JANUARY: 0.80,
                MARCH: 0.90,
                JUNE: 1.0,
                JULY: 1.0,
                AUGUST: 1.21,
                SEPTEMBER: 1.14,
                OCTOBER: 0.85,
                DECEMBER: 0.82,
            }
            
            seasonal1 = YearlySeasonality(constraints, normalize=True)
            
            
            # combine all patterns and merge
            
            # unique dates
            idx = pd.DataFrame(dts, columns=['date'])
            switch_pattern = pd.DataFrame(realistic_speed.generate(time_points=list(pd.to_datetime(dts))), columns=['switch1'])
            switch_pattern2 = pd.DataFrame(realistic_speed2.generate(time_points=list(pd.to_datetime(dts))), columns=['switch2'])
            seasonal_pattern = pd.DataFrame(seasonal1.generate(time_points=list(pd.to_datetime(dts))), columns=['annual'])
            
            # combine patterns
            pattern = pd.concat([idx, switch_pattern, switch_pattern2, seasonal_pattern], axis=1)
            
            # multiply patterns into 1
            pattern[result_col_name] = pattern['switch1'] * pattern['switch2'] * pattern['annual']
            
            # merge original with the pattern combo
            new = original.merge(pattern, how='inner', on=['date'])

            # assemble into single result
            if len(func_end_result)==0:
                func_end_result = new
            else:
                func_end_result = pd.concat([func_end_result,new],axis=0)
                
    return func_end_result

###################################

# TODO: do this better
# add helpers to apply patterns to different things
base2 = add_helper_series(merged, 'helper_impressions')
base3 = add_helper_series(base2, 'helper_clicks')
base4 = add_helper_series(base3, 'helper_purchases')
final = base4

# overwrite the random values with the value pattern

# impressions 
final['impressions'] = (final['helper_impressions'] * final['impressions']).round()

# TODO: do this better?

# trying to artificially penalize clicks and purchaes for campaigns with higher impressions
# to simulate saturation
# using (2 - log(x)/20) just because it felt okay
final['clicks'] = ((final['clicks_pct'] * final['helper_clicks']) * final['impressions'] * (2 - (np.log2(final['impressions']+1)/20))).round()
final['purchases'] = (final['purchases_pct'] * final['helper_purchases'] * final['clicks'] * (2 - (np.log2(final['impressions']+1)/20))).round()

# revenue 
final['revenue'] = (final['rev_per'] * final['purchases']).round()

# TODO: do this better?

# trying to simulate higher cost for campaigns with higher impressions
# not realistic but it'll help by making profit go down
# using (1 + (og(impressions^2))/100 because it felt okay
final['cost_multiplier'] = np.where(final['impressions']>=10000, 2.0, 1.0)
final['cost'] = ((((final['cost_per']/60) * final['cost_multiplier'] * final['impressions']))* (1 + (np.log(final['impressions']**2))/100)).round()

# only keep certain columns
final = final[['date','platform','tactic','campaign_name','impressions','clicks','purchases','revenue','cost']].copy()

# do a basic visualization
for p in final['platform'].unique():
    grouped = final[final['platform']==p].groupby('date')['impressions']
    result = grouped.sum()
    result.plot()
