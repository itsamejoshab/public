from pathlib import Path
import os
import numpy as np
import pandas as pd

import random

# display options for notebooks only
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 25)

# set path directories
curr_dir = Path(os.getcwd())
print('Current Directory is: ', str(curr_dir))
data_dir = Path(curr_dir.parents[0] / 'data/')
artifacts_dir = Path(curr_dir / 'artifacts/')

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

#########################################################################################

seed = 20171230
np.random.seed(seed)

TARGET_VARIABLE_NAME = 'target'
SEGMENT_TARGET_PCT = 0.1178
CUSTOMER_TYPE = 'Residential'
SNAPSHOT_DATE = pd.to_datetime('2020-05-31')
output_filename = 'sas_fakemodel_testdata.csv'
n_rows = 10000
NOISE_FACTOR = 45 #45

country_list = ['ME', 'VT', 'NH', 'MA', 'CT', 'RI']
country_list_dist = [0.09, 0.05, 0.31, 0.26, 0.27, 0.02]
country_list_map = {'ME':0.98, 'VT':1.08, 'NH':0.995, 'MA':1.05, 'CT':0.97, 'RI':1.03}

product_mix_list = ['Residential Direct','Solar Rebate Capable','Commercial Charitable']
product_mix_list_dist = [0.95, 0.04, 0.01]
product_mix_list_map = {'Residential Direct':1.0,'Solar Rebate Capable':1.1, 'Commercial Charitable': 0.22}

division_list = ['North','South','East','West']
division_list_dist = [0.19, 0.26, 0.26, 0.29]
division_list_map = {'North':0.99, 'East':0.98, 'West':1.00, 'South':1.01}

industry_list = ['Cheap Charlie','Fresh Frank','Jealous Jenny','Cranky Chris','Loving Larry','Serious Sarah','Friendly Fred','Hilarious Hillary']
industry_list_dist = [0.11, 0.08, 0.05, 0.14, 0.08, 0.06, 0.19, 0.29]
industry_list_map = {'Cheap Charlie':1.01,'Fresh Frank':0.96,'Jealous Jenny':0.99,'Cranky Chris':0.95,'Loving Larry':1.00,'Serious Sarah':1.00,'Friendly Fred':0.99,'Hilarious Hillary':0.97}

# TEXT VARIABLES
# words to be used to generate random sentences. Some trigger a target=0 and some trigger a target=1
vocab_neutral = ['customer wanted to check on something', 'pay their bill online', 'checked online', 'forgot password', 'tree trimming', 'very windy', 'powerlines swaying in the wind', 
                 'statement', 'did not check email', 'contract', 'promotion', 'customer','account','fine','okay','got a letter','would like a paper copy', 'paper copy of the bill','schedule a visit','visit from a technician',
                'come trim the trees','inquiry about solar program','wants to know what appliance to purchase','energy','energy','efficient','efficiency','promises to','give us a call','whenever','neighbor','their']
vocab_trigger_0 = ['lost job','got fired','moved','abandoned property','vacant','died','passed away','arrested','in jail','incarcerated','covid']
vocab_trigger_1 = ['going on vacation','traveling outside of the country','temporary','heads-up','just changed bank accounts','credit-card expired','new credit-card number','forgot','forget',
                  'just forgot','forgot to enroll in autopayment','auto-pay']

#! Generate new dataframe
df = pd.DataFrame(np.random.randint(0, 2147483647, size=(n_rows, 1)).astype('str'), columns=['AcctID'])
df['AcctID'] = 'C' + df['AcctID']
df['snapshot_date'] = SNAPSHOT_DATE
#### Start simulating Columns
df['customer_type'] = CUSTOMER_TYPE

# Categorical variables
df['division'] = np.random.choice(division_list, df.shape[0], p=division_list_dist)
df['state'] = np.random.choice(country_list, df.shape[0], p=country_list_dist)
df['product'] = np.random.choice(product_mix_list, df.shape[0], p=product_mix_list_dist)
df['marketing_segment'] = np.random.choice(industry_list, df.shape[0], p=industry_list_dist)

# Numeric - Simple gamma distributions
df['estimated_annual_income'] = np.round(np.random.gamma(5, 85, n_rows), decimals=0).astype('int64')*100
df['months_since_account_begin'] = np.round(np.random.gamma(5, 4.2, n_rows), decimals=0).astype('int64')
df['__tenure_knot_strong'] = np.where(np.mod(df['months_since_account_begin'], 36)==0, 1, 0)
df['__tenure_knot_weak'] = np.where(np.mod(df['months_since_account_begin'], 12)==0, 1, 0)
df['estimated_age'] = np.round(np.random.gamma(30, 1.5, n_rows),0)
df['num_emails_opened_last6m'] = np.random.poisson(0.5, n_rows)

# Numeric - Simple beta distributions
df['year_home_built'] = np.round(2020 - np.random.beta(3, 1.5, n_rows) * 65,0)

# Counts (poisson)
df['times_called_in_last_month'] = np.random.poisson(1, n_rows)
df['times_emailed_support_in_last_month'] = np.random.poisson(0.6,n_rows)
df['number_outages_last1year'] = np.round(1*np.random.gamma(1, 5.5, n_rows), decimals=0).astype('int64')

# binary
df['service_contract_ind'] = np.random.binomial(1, 0.02, n_rows) 
df['home_own_rent'] = np.random.binomial(1, 0.65, n_rows)
df['solar_rebate_ind'] = np.random.binomial(1, 0.02, n_rows) 
df['underground_drop_ind'] = np.random.binomial(1, 0.95, n_rows) 
df['ebill_ind'] = np.random.binomial(1, 0.15, n_rows) 
df['has_mobile'] = np.random.binomial(1, 0.14, n_rows) 
df['past_due_ind'] = 1

# Bi modal distrubution
df['__dist1'] = np.round((np.random.gamma(3, 0.3, n_rows)+0.5) * 25, decimals=6).astype('float64') # one distribution
df['__dist2'] = np.random.beta(3, 1.5, n_rows) * 150 # another distribution
df['__5050'] = np.round(np.random.binomial(1, 0.5, n_rows).astype('float'), 2)  # 50-50 chance 
df['avg_monthly_bill'] = np.round(((df['__dist1'] * df['__5050']) + ((1 - df['__5050']) * df['__dist2'])) , 2)

# dependent on others
df['avg_kwh_usage_last1year'] = (df['avg_monthly_bill'] / 0.47) * np.random.beta(8, 1, n_rows)
df['called_in_last_month'] = np.where(df.times_called_in_last_month>0, 1, 0)
df['num_distinct_outages_l90d'] = np.random.binomial(df.number_outages_last1year.astype('int64'), .5, n_rows) * np.random.beta(8, 1, n_rows)
df['__diceroll'] = np.round(np.random.binomial(1, 0.2, n_rows).astype('float'), 2)  
df['30_day_balance'] = np.round(((df['avg_monthly_bill'] * df['__diceroll']) + ((1 - df['__diceroll']) * (df['avg_monthly_bill'] * np.random.beta(3, 1.5, n_rows)))) , 2)
df['60_day_balance'] = np.round(((df['30_day_balance'] * df['__diceroll']) + ((1 - df['__diceroll']) * (df['30_day_balance']))) , 2)
df['90_day_balance'] = np.round(((df['60_day_balance'] * df['__diceroll']) + ((1 - df['__diceroll']) * (df['60_day_balance']))) , 2)
df['120_day_balance'] = np.round(((df['90_day_balance'] * df['__diceroll']) + ((1 - df['__diceroll']) * (df['90_day_balance']))) , 2)
df['delinquency_status'] = np.where(df['120_day_balance'] > 0, '120 days',
                                   np.where(df['90_day_balance'] > 0, '90 days',
                                           np.where(df['60_day_balance']>0, '60 days','30 days')))
delinq_map = {'120 days':0.6,'90 days':0.8, '60 days': 0.9, '30 days': 1.0}




########################################
# Make everything numeric and scale

#! Create temp_ copies of all numeric columns
nums =  [c for c in df._get_numeric_data().columns if 'temp' not in c]

cols = ['temp_' + c for c in nums]

df_scaled = df.copy()

from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()
df_scaled[nums] = scaler.fit_transform(df_scaled[nums])
for i, c in enumerate(nums, start=0):
    df_scaled.rename(columns = {nums[i]:cols[i]}, inplace = True) 
    
df_scaled.filter(regex='temp')

df = pd.concat([df, df_scaled.filter(regex='temp')], axis=1, sort=False)

# dont forget the categorical ones (make them numeric too)
# it sucks that I cant make this work in the loop scaler-code above
# , but Ive wasted enough time on this day 
df['temp_division'] = df['division'].map(division_list_map)
df['temp_state'] = df['state'].map(country_list_map)
df['temp_product'] = df['product'].map(product_mix_list_map)
df['temp_marketing_segment'] = df['marketing_segment'].map(industry_list_map)
df['temp_delinquency_status'] = df['delinquency_status'].map(delinq_map)

# Bathtub relationships
for col in ['avg_monthly_bill','estimated_annual_income','months_since_account_begin']:
    df['__std'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    df['__scaled'] = df['__std'] * (np.pi - -np.pi) + -np.pi
    df['temp_' + col] = np.tanh(df['__scaled'])

#########################################
#### fake propensity (this formula will refer to the temp_ versions!)
df['__logit'] = (
                             (   # manually control these:
                                 0.05 * df['temp_estimated_annual_income'] + \
                                 0.2 * df['temp_months_since_account_begin'] + \
                                 0.2 * df['temp_estimated_age'] + \
                                 0.15 * df['temp_num_emails_opened_last6m'] + \
                                -0.1 * df['temp_year_home_built'] + \
                                -0.005 * df['temp_times_called_in_last_month'] + \
                                -0.005 * df['temp_times_emailed_support_in_last_month'] + \
                                -0.1 * df['__tenure_knot_strong'] + \
                                -0.1 * df['__tenure_knot_weak'] + \
                                -0.05 * df['temp_number_outages_last1year'] + \
                                 0.2 * df['temp_service_contract_ind'] + \
                                 0.05 * df['temp_home_own_rent'] + \
                                 0.1 * df['temp_solar_rebate_ind'] + \
                                 0.005 * df['temp_underground_drop_ind'] + \
                                 0.05 * df['temp_ebill_ind'] + \
                                 0.05 * df['temp_has_mobile'] + \
                                 0.2 * df['temp_avg_monthly_bill'] + \
                                -0.1 * df['temp_avg_kwh_usage_last1year'] + \
                                -0.1 * df['temp_30_day_balance'] + \
                                -0.05 * df['temp_60_day_balance'] + \
                                -0.05 * df['temp_90_day_balance'] + \
                                -0.1 * df['temp_120_day_balance'] + \
                                 
                                 
                                 # linear effects - optional
                                 0 +
                                 # weird interactions - I use a script to randomly generate these
                                 -0.042 *(df['temp_60_day_balance'] * df['temp_months_since_account_begin']) + 0.02 *(df['temp_estimated_annual_income'] * np.sin(df['temp_underground_drop_ind'])) + 0.073 *(np.exp(df['temp_estimated_annual_income']) * np.cos(df['temp_home_own_rent'])) + -0.05 *(df['temp_months_since_account_begin'] * np.sin(df['temp_60_day_balance'])) + 0.022 *(np.sin(df['temp_months_since_account_begin']) * np.sin(df['temp_times_called_in_last_month']))
                              ) * (df.temp_division * df.temp_state * df.temp_product * df.temp_marketing_segment * df.temp_delinquency_status)
                          ) 
                            # sigmoid transformation not really required
df['__MLpropensity'] = 1 / (1 + np.exp(-df['__logit']))


    #####################################
# first make a function to create a random string
def random_string_from_list(list_of_words):
    
    # First randomized the number of words to grab (in list)
    random_words_list = []
    number_of_words = np.round(np.random.gamma(4,1,1) + 1, decimals=0)[0].astype('int')
    #Loop through and append to list, based on how many words we are sampling
    for i in range(number_of_words):
        random_index = np.random.randint(0,len(list_of_words))
        random_words_list.append(list_of_words[random_index])
    
    #now convert the list of random words to a string
    random_string = ' '.join(random_words_list)
    
    return(random_string)

# add some more variables now that we have a propensity calculated
df['csrNotes'] = [random_string_from_list(vocab_neutral) for x in range(df.shape[0])]
df['__csrNotes_trigger0'] = pd.Series([random.choice(list(vocab_neutral + vocab_trigger_0)) for _ in range(df.shape[0])])
df['__csrNotes_trigger1'] = pd.Series([random.choice(list(vocab_neutral + vocab_trigger_1)) for _ in range(df.shape[0])])
df['csrNotes'] = np.where(df.__MLpropensity >= df.__MLpropensity.quantile(q=0.7) 
                          , df.csrNotes + ' ' + df.__csrNotes_trigger1
                                , np.where(df.__MLpropensity < df.__MLpropensity.quantile(q=0.1)
                                           , df.csrNotes + ' ' + df.__csrNotes_trigger0
                                                , df.csrNotes))


# lets add noise before we make the knowns
maxx = df['__MLpropensity'].quantile(q=0.95)
minn = df['__MLpropensity'].quantile(q=0.05)
scale = (maxx-minn) / 100

df['__noise'] = np.random.normal(0,NOISE_FACTOR,n_rows) * scale
df['__MLpropensity'] = df['__MLpropensity'] + df['__noise']


#! target definition (error should already be baked into this!)
df[TARGET_VARIABLE_NAME] = np.where(df.__MLpropensity > df.__MLpropensity.quantile(q=(1-SEGMENT_TARGET_PCT)), 1, 0)
#Fdf['demonstration_column'] = pd.qcut(df['__MLpropensity'], 3, labels=[3,2,1])

# introduce a slip
df['__slip'] = np.random.randint(0,len(df),size=len(df))
SLIP_AMT = SEGMENT_TARGET_PCT / 100
df[TARGET_VARIABLE_NAME] = np.where(df.__slip < df.__slip.quantile(q=(SLIP_AMT/2)), np.where(df[TARGET_VARIABLE_NAME]==1, 0, 1), df[TARGET_VARIABLE_NAME])
df[TARGET_VARIABLE_NAME] = np.where(df.__slip > df.__slip.quantile(q=(1-(SLIP_AMT/2))), np.where(df[TARGET_VARIABLE_NAME]==0, 1, 0), df[TARGET_VARIABLE_NAME])

# include some missing values
# Be careful here - replacing int64 with '' gives weird errors
for col in df.columns:
    if True == True and col not in ['AcctID','snapshot_date','customer_type','division','state','product','marketing_segment','months_since_account_begin','num_emails_opened_last6m','times_called_in_last_month', \
                                     'times_emailed_support_in_last_month','number_outages_last1year','service_contract_ind','solar_rebate_ind','underground_drop_ind','ebill_ind','has_mobile','past_due_ind','avg_monthly_bill', \
                                    'avg_kwh_usage_last1year','called_in_last_month','num_distinct_outages_l90d','30_day_balance','60_day_balance','90_day_balance','120_day_balance','delinquency_status','target',
                                      '__MLpropensity','__logit']:
        df[col] = np.where(np.random.binomial(1, 0.73, n_rows) >= 1, df[col], np.nan)

# Map some variables back to text for some realism

ysn_map = {'1':'Yes','0':'No'}
ownrent_map = {'1.0': 'Own', '0.0': 'Rent'}
bill_map = {'1': 'Electronic Billing', '0': 'Paper Bill'}

for col in ['service_contract_ind','solar_rebate_ind','home_own_rent','ebill_ind']:
    
    if col in ['service_contract_ind','solar_rebate_ind']:
        mymap = ysn_map
    elif col in ['home_own_rent']:
        mymap = ownrent_map
    elif col in ['ebill_ind']:
        mymap = bill_map
        
    df['temp_' + col] = df[col].astype('str')
    df.drop(columns=[col],inplace=True)
    df[col] = df['temp_' + col].map(mymap)
    df.drop(columns=['temp_' + col],inplace=True)
    
# drop helper columns
df = df[df.columns.drop(list(df.filter(regex='__')))]
df = df[df.columns.drop(list(df.filter(regex='temp')))]

df.head(6)
