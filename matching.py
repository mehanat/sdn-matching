import re
import nltk
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction
smoothie = SmoothingFunction().method4
from rapidfuzz.distance import Levenshtein, Jaro, JaroWinkler
from rapidfuzz.fuzz import ratio
import numpy as np
import pandas as pd

#import modin.pandas as pd

import time
import math

import itertools


nonalphanumeric_re = re.compile(r"[^\w ]+")
whitespace_re = re.compile(r" +")

def seconds_to_str(secs):
    if secs >= 3600:
        return time.strftime('%H:%M:%S', time.gmtime(secs))
    else:
        return time.strftime('%M:%S', time.gmtime(secs))

def normalize_simple(name: str) -> str:
    return whitespace_re.sub(" ", nonalphanumeric_re.sub(" ", name.lower()))

def compare_names(full_name_client, full_name_sdn, name_matching_method, **kwargs):
    full_name_client_norm = normalize_simple(full_name_client)
    full_name_sdn_norm = normalize_simple(full_name_sdn)
    
    if name_matching_method == 'bleu':
        score = bleu([full_name_client_norm], full_name_sdn_norm, smoothing_function=smoothie)
        return 100 * round(score, 3), 1000 * round(score, 2)
    elif name_matching_method == 'levenshtein':
        score = Levenshtein.normalized_similarity(full_name_client_norm, full_name_sdn_norm)
        return 100 * round(score, 3), 1000 * round(score, 2)
    elif name_matching_method == 'jaro_winkler':
        score = JaroWinkler.similarity(full_name_client_norm, full_name_sdn_norm)
        return 100 * round(score, 3), 1000 * round(score, 2)
    elif name_matching_method == 'fuzz':
        score = ratio(full_name_client_norm, full_name_sdn_norm)
        return round(score, 3), 10 * round(score, 1)
    else:
        raise TypeError(f'"{name_matching_method}" is an invalid keyword argument for compare_names()')

def compare_addresses(res, address_matching_method, **kwargs):
    try:
        address_client_norm = normalize_simple(res['address_x'])
        address_sdn_norm = normalize_simple(res['address_y'])
        city_client_norm = normalize_simple(res['city_x'])
        city_sdn_norm = normalize_simple(res['city_y'])
        country_client_norm = normalize_simple(res['country_x'])
        country_sdn_norm = normalize_simple(res['country_y'])

        if address_matching_method == 'bleu':
            score_address = bleu([address_client_norm], address_sdn_norm, smoothing_function=smoothie)
            score_city = bleu([city_client_norm], city_sdn_norm, smoothing_function=smoothie)
            score_country = bleu([country_client_norm], country_sdn_norm, smoothing_function=smoothie)
        elif address_matching_method == 'levenshtein':
            score_address = Levenshtein.normalized_similarity(address_client_norm, address_sdn_norm)
            score_city = Levenshtein.normalized_similarity(city_client_norm, city_sdn_norm)
            score_country = Levenshtein.normalized_similarity(country_client_norm, country_sdn_norm)
        else:
            raise TypeError(f'"{address_matching_method}" is an invalid keyword argument for compare_addresses()')
        
        return 200*round(score_address, 2) + 100*round(score_city, 2) + (400*round(score_country, 2)-200)
    except AttributeError:
        return 0


def compare_names_with_progress(full_name_client, full_name_sdn, name_matching_method, total, start):
    global i
    if i % 5000 == 0:
        s = i/total
        sec_passed = time.perf_counter() - start
        sec_remaining = (sec_passed / s) * (1 - s)
        print(f"{i}/{total} ({s * 100:.1f}%)  Time passed={seconds_to_str(sec_passed)}  ; Time remaining={seconds_to_str(sec_remaining)} ")
    return compare_names(full_name_client, full_name_sdn, name_matching_method=name_matching_method)


def scoring_name(listFromClient, sanctionData, name_matching_method, sep):
    names_client = listFromClient.assign(matching_name_client=listFromClient['Name'].str.split(sep)).explode('matching_name_client')
    names_sdn = sanctionData.assign(matching_name_sdn=sanctionData['Name'].str.split(sep)).explode('matching_name_sdn')

    res = pd.merge(names_client, names_sdn, how='cross')

    res['Name_Matching_%'], res['score_Name'] = zip(*res.apply(lambda x: compare_names(x["matching_name_client"], x["matching_name_sdn"], name_matching_method=name_matching_method), axis=1))


    res = res.rename(columns={'Name_x': 'name_client(multi_value)',

                                          'Name_y': 'name_sdn(multi_value)'})

    res = res[['id_client', 'id_sdn', 'name_client(multi_value)', 'name_sdn(multi_value)', 'matching_name_client', 

                           'matching_name_sdn', 'Name_Matching_%', 'score_Name']]

    return res.loc[res.groupby(['id_client', 'id_sdn'])['score_Name'].idxmax()]


def scoring_dob(listFromClient, sanctionData, sep):
    dob_client = listFromClient.assign(matching_DoB_client=listFromClient['DoB'].str.split(sep)).explode('matching_DoB_client')
    dob_sdn = sanctionData.assign(matching_DoB_sdn=sanctionData['DoB'].str.split(sep)).explode('matching_DoB_sdn')

    dob_sdn['matching_DoB_sdn'] = dob_sdn['matching_DoB_sdn'].str.replace('circa', 'to')
    dob_sdn['isRange'] = dob_sdn['matching_DoB_sdn'].str.contains('to').fillna(False)
    dob_sdn[['fromDoB', 'toDoB']] = dob_sdn['matching_DoB_sdn'].str.split('to ', n=1, expand=True).reindex(range(2), axis=1)
            
    dob_sdn['fromDoB'] = pd.to_datetime(dob_sdn['fromDoB'], infer_datetime_format=True)
    dob_sdn['toDoB'] = pd.to_datetime(dob_sdn['toDoB'], infer_datetime_format=True)
    dob_client['DoB_client'] = pd.to_datetime(dob_client['matching_DoB_client'], infer_datetime_format=True)

    res = pd.merge(dob_client, dob_sdn, how='cross')

    res['l_year']   = res['DoB_client'].dt.year
    res['l_month']  = res['DoB_client'].dt.month
    res['l_day']    = res['DoB_client'].dt.day
    res['rf_year']  = res['fromDoB'].dt.year
    res['rf_month'] = res['fromDoB'].dt.month
    res['rf_day']   = res['fromDoB'].dt.day
    res['rt_year']  = res['toDoB'].dt.year
    res['rt_month'] = res['toDoB'].dt.month
    res['rt_day']   = res['toDoB'].dt.day
            
            # Scoring
    res['scored'] = False
    fields = ['score_DoB', 'scored']
            # Case when DoB in sanction data is not a range ===
            ## 100% DoB match
    res.loc[(res['DoB_client'] == res['fromDoB']) &
                    ~res['isRange'] & 
                    ~res['scored'], fields] = 400, True 
            ## Year and Month match only
    res.loc[(res['l_year'] == res['rf_year']) & 
                    (res['l_month'] == res['rf_month']) &
                    ~res['isRange'] &
                    ~res['scored'], fields] = 250, True
            ## Year match only
    res.loc[(res['l_year'] == res['rf_year']) &
                    ~res['isRange'] &
                    ~res['scored'], fields] = 200, True
            ## Dates within 2 years of each other
    res.loc[(np.abs(res['fromDoB'] - res['DoB_client']) /  np.timedelta64(1, 'Y') < 2) &
                    ~res['isRange'] &
                    ~res['scored'], fields] = 100, True
            ## More than 2 years difference in DoB between bank and World-Check’ information
    res.loc[(np.abs(res['fromDoB'] - res['DoB_client']) /  np.timedelta64(1, 'Y') >= 2) &
                    ~res['isRange'] &
                    ~res['scored'], fields] = -200, True
            ## Date of Birth not recorded on Bank or World-Check’ list
    res.loc[~res['isRange'] & 
                    ~res['scored'], fields] = 0, True 
            # === Case when DoB in sanction data is a range ===
            ## Date is before ('circa' case)
    res.loc[(res['DoB_client'] <= res['toDoB']) &
                    (res['fromDoB'].isna()) &
                    ~res['scored'], fields] = 100, True
            ## Date is in range
    res.loc[(res['DoB_client'] <= res['toDoB']) &
                    (res['fromDoB'] <= res['DoB_client']) &
                    ~res['scored'], fields] = 150, True
            ## Date is not in range
    res.loc[~res['scored'], fields] = 0, True


    res = res.rename(columns={'DoB_x': 'DoB_client(multi_value)',
                                          'DoB_y': 'DoB_sdn(multi_value)'})
    res = res[['id_client', 'id_sdn', 'DoB_client(multi_value)', 'DoB_sdn(multi_value)', 'matching_DoB_client', 
                           'matching_DoB_sdn', 'score_DoB']]

    return res.loc[res.groupby(['id_client', 'id_sdn'])['score_DoB'].idxmax()]

def scoring_address(listFromClient, sanctionData, sep):
    address_client = listFromClient.assign(matching_address_client=listFromClient['Address'].str.split(sep)).explode('matching_address_client')
    address_client = address_client.assign(matching_country_client=address_client['Country'].str.split(sep)).explode('matching_country_client')
    address_sdn = sanctionData.assign(matching_address_sdn=sanctionData['Address'].str.split(sep)).explode('matching_address_sdn')
    address_sdn = address_sdn.assign(matching_country_sdn=address_sdn['Country'].str.split(sep)).explode('matching_country_sdn')

    res = pd.merge(address_client, address_sdn, how='cross')

            #Scoring addresses
    res['scored'] = False
    fields = ['score_Address', 'scored']
            #Exact math of address and country
    res.loc[(res['matching_address_client'] == res['matching_address_sdn']) &
                    (res['matching_country_client'] == res['matching_country_sdn']) &
                    ~res['scored'], fields] = 500, True
            #Match of address only
    res.loc[(res['matching_address_client'] == res['matching_address_sdn']) &
                    ~res['scored'], fields] = 200, True
            #Match of country only
    res.loc[(res['matching_country_client'] == res['matching_country_sdn']) &
                    ~res['scored'], fields] = 300, True
            #No data
    res.loc[(res['matching_country_client'].isna()) |
                    (res['matching_country_sdn'].isna()) &
                    ~res['scored'], fields] = 0, True
            #No match
    res.loc[~res['scored'], fields] = -200, True

            
    res = res.rename(columns={'Address_x': 'Address_client(multi_value)',
                                          'Address_y': 'Address_sdn(multi_value)',
                                          'Country_x': 'Country_client(multi_value)',
                                          'Country_y': 'Country_sdn(multi_value)'})
    res = res[['id_client', 'id_sdn', 'Address_client(multi_value)', 'Address_sdn(multi_value)', 'matching_address_client', 
                           'matching_address_sdn', 'Country_client(multi_value)', 'Country_sdn(multi_value)', 'matching_country_client',
                           'matching_country_sdn', 'score_Address']]

    return res.loc[res.groupby(['id_client', 'id_sdn'])['score_Address'].idxmax()]

def score_transaction(listFromClient, sanctionData, name_matching_method, address_matching_method, scoreTrsh, mini_batch, sep):
    res_score = pd.DataFrame()
    start = time.perf_counter()
    i = 0
    total = math.ceil(len(sanctionData) / mini_batch) * math.ceil(len(listFromClient) / mini_batch)

    sanctionData = sanctionData.rename(columns={'full_name_sdn': 'Name', 'uid': 'id_sdn', 'address':'Address', 'city':'City', 'country':'Country'})
    listFromClient = listFromClient.rename(columns={'UniqueID': 'id_client'})

    print(f"{i}/{total} (0%) ")
    for batch_sanct in range(0, len(sanctionData), mini_batch):
        for batch_cl in range(0, len(listFromClient), mini_batch):
            sanctionData_sub = sanctionData[batch_sanct:batch_sanct+mini_batch]
            listFromClient_sub = listFromClient[batch_cl:batch_cl+mini_batch]

            client_names_data = listFromClient_sub[['id_client', 'Name']]
            sdn_names_data = sanctionData_sub[['id_sdn', 'Name']]

            client_addresses_data = listFromClient_sub[['id_client', 'Address', 'Country']]
            sdn_addresses_data = sanctionData_sub[['id_sdn', 'Address', 'City', 'Country']]

            names_score = scoring_name(client_names_data, sdn_names_data, name_matching_method = name_matching_method, sep = sep)
            #print('names_score', names_score.shape)

            addresses_scores = scoring_address(client_addresses_data, sdn_addresses_data, sep = sep)
            #print('addresses_scores', names_score.shape)

            res = pd.merge(names_score, addresses_scores, how='outer', on = ['id_client', 'id_sdn'])

            res[['DoB', 'fromDoB', 'score_DoB']] = np.NaN

            res['Overall_Score'] = res['score_Name'] + res['score_Address']

            res = res.query(f'Overall_Score >= {scoreTrsh}')
            res_score = pd.concat([res_score, res])

            i+=1
            s = i/total
            sec_passed = time.perf_counter() - start
            sec_remaining = (sec_passed / s) * (1 - s)
            print(f"{i}/{total} ({s * 100:.1f}%)  Time passed={seconds_to_str(sec_passed)}  ; Time remaining={seconds_to_str(sec_remaining)} ")
    res_score = res_score.drop_duplicates()
    return res_score

def score_physical(listFromClient, sanctionData, name_matching_method, address_matching_method, scoreTrsh, mini_batch, sep):
    res_score = pd.DataFrame()
    start = time.perf_counter()
    i = 0
    total = math.ceil(len(sanctionData) / mini_batch) * math.ceil(len(listFromClient) / mini_batch)

    sanctionData = sanctionData.rename(columns={'full_name_sdn': 'Name', 'uid': 'id_sdn', 'address':'Address', 'city':'City', 'country':'Country', 'dateOfBirth':'DoB'})
    listFromClient = listFromClient.rename(columns={'UniqueID': 'id_client', 'DateOfBirth':'DoB'})

    print(f"{i}/{total} (0%) ")
    for batch_sanct in range(0, len(sanctionData), mini_batch):
        for batch_cl in range(0, len(listFromClient), mini_batch):
            sanctionData_sub = sanctionData[batch_sanct:batch_sanct+mini_batch]
            listFromClient_sub = listFromClient[batch_cl:batch_cl+mini_batch]

            client_names_data = listFromClient_sub[['id_client', 'Name']]
            sdn_names_data = sanctionData_sub[['id_sdn', 'Name']]

            client_addresses_data = listFromClient_sub[['id_client', 'Address', 'Country']]
            sdn_addresses_data = sanctionData_sub[['id_sdn', 'Address', 'City', 'Country']]

            names_score = scoring_name(client_names_data, sdn_names_data, name_matching_method = name_matching_method, sep = sep)
            #print('names_score', names_score.shape)

            addresses_scores = scoring_address(client_addresses_data, sdn_addresses_data, sep = sep)
            #print('addresses_scores', names_score.shape)

            client_dates_data = listFromClient_sub[['id_client', 'DoB']]
            sdn_dates_data = sanctionData_sub[['id_sdn', 'DoB']]

            dob_scores = scoring_dob(client_dates_data, sdn_dates_data, sep = sep)

            res = pd.merge(names_score, addresses_scores, how='outer', on = ['id_client', 'id_sdn'])
            res = pd.merge(res, dob_scores, how='outer', on = ['id_client', 'id_sdn'])

            res[['DoB', 'fromDoB', 'score_DoB']] = np.NaN

            res['Overall_Score'] = res['score_Name'] + res['score_Address'] + res['score_DoB']

            res = res.query(f'Overall_Score >= {scoreTrsh}')
            res_score = pd.concat([res_score, res])

            i+=1
            s = i/total
            sec_passed = time.perf_counter() - start
            sec_remaining = (sec_passed / s) * (1 - s)
            print(f"{i}/{total} ({s * 100:.1f}%)  Time passed={seconds_to_str(sec_passed)}  ; Time remaining={seconds_to_str(sec_remaining)} ")
    res_score = res_score.drop_duplicates()
    return res_score

def score_moral(listFromClient, sanctionData, name_matching_method, address_matching_method, scoreTrsh, mini_batch, sep):
    res_score = pd.DataFrame()
    start = time.perf_counter()
    i = 0
    total = math.ceil(len(sanctionData) / mini_batch) * math.ceil(len(listFromClient) / mini_batch)

    sanctionData = sanctionData.rename(columns={'full_name_sdn': 'Name', 'uid': 'id_sdn', 'address':'Address', 'city':'City', 'country':'Country'})
    listFromClient = listFromClient.rename(columns={'UniqueID': 'id_client'})
    print(f"{i}/{total} (0%) ")
    for batch_sanct in range(0, len(sanctionData), mini_batch):
        for batch_cl in range(0, len(listFromClient), mini_batch):
            sanctionData_sub = sanctionData[batch_sanct:batch_sanct+mini_batch]
            listFromClient_sub = listFromClient[batch_cl:batch_cl+mini_batch]

            client_names_data = listFromClient_sub[['id_client', 'Name']]
            sdn_names_data = sanctionData_sub[['id_sdn', 'Name']]

            client_addresses_data = listFromClient_sub[['id_client', 'Address', 'Country']]
            sdn_addresses_data = sanctionData_sub[['id_sdn', 'Address', 'City', 'Country']]

            names_score = scoring_name(client_names_data, sdn_names_data, name_matching_method = name_matching_method, sep = sep)
            #print('names_score', names_score.shape)

            addresses_scores = scoring_address(client_addresses_data, sdn_addresses_data, sep = sep)
            #print('addresses_scores', names_score.shape)

            res = pd.merge(names_score, addresses_scores, how='outer', on = ['id_client', 'id_sdn'])

            res[['DoB', 'fromDoB', 'score_DoB']] = np.NaN

            res['Overall_Score'] = res['score_Name'] + res['score_Address']

            res = res.query(f'Overall_Score >= {scoreTrsh}')
            res_score = pd.concat([res_score, res])

            i+=1
            s = i/total
            sec_passed = time.perf_counter() - start
            sec_remaining = (sec_passed / s) * (1 - s)
            print(f"{i}/{total} ({s * 100:.1f}%)  Time passed={seconds_to_str(sec_passed)}  ; Time remaining={seconds_to_str(sec_remaining)} ")
    res_score = res_score.drop_duplicates()
    return res_score

def score_customers(listFromClient, sanctionData, name_matching_method, address_matching_method, scoreTrsh, mini_batch, sep):
    res_score = pd.DataFrame()
    start = time.perf_counter()
    i = 0
    total = math.ceil(len(sanctionData) / mini_batch) * math.ceil(len(listFromClient) / mini_batch)

    sanctionData = sanctionData.rename(columns={'full_name_sdn': 'Name', 'uid': 'id_sdn', 'address':'Address', 'city':'City', 'country':'Country'})
    listFromClient = listFromClient.rename(columns={'UniqueID': 'id_client'})
    print(f"{i}/{total} (0%) ")
    for batch_sanct in range(0, len(sanctionData), mini_batch):
        for batch_cl in range(0, len(listFromClient), mini_batch):
            sanctionData_sub = sanctionData[batch_sanct:batch_sanct+mini_batch]
            listFromClient_sub = listFromClient[batch_cl:batch_cl+mini_batch]

            client_names_data = listFromClient_sub[['id_client', 'Name']]
            sdn_names_data = sanctionData_sub[['id_sdn', 'Name']]

            client_addresses_data = listFromClient_sub[['id_client', 'Address', 'Country']]
            sdn_addresses_data = sanctionData_sub[['id_sdn', 'Address', 'City', 'Country']]

            names_score = scoring_name(client_names_data, sdn_names_data, name_matching_method = name_matching_method, sep = sep)
            #print('names_score', names_score.shape)

            addresses_scores = scoring_address(client_addresses_data, sdn_addresses_data, sep = sep)
            #print('addresses_scores', names_score.shape)

            res = pd.merge(names_score, addresses_scores, how='outer', on = ['id_client', 'id_sdn'])

            res[['DoB', 'fromDoB', 'score_DoB']] = np.NaN

            res['Overall_Score'] = res['score_Name'] + res['score_Address']

            res = res.query(f'Overall_Score >= {scoreTrsh}')
            res_score = pd.concat([res_score, res])

            i+=1
            s = i/total
            sec_passed = time.perf_counter() - start
            sec_remaining = (sec_passed / s) * (1 - s)
            print(f"{i}/{total} ({s * 100:.1f}%)  Time passed={seconds_to_str(sec_passed)}  ; Time remaining={seconds_to_str(sec_remaining)} ")
    res_score = res_score.drop_duplicates()
    return res_score

def score(listFromClient, sanctionData, input_type, name_matching_method='bleu', address_matching_method = 'exact_match', mini_batch=10000, scoreTrsh = 800, sep='¦'):
    #todo
    sanctionData['sanction_list_name'] = 'SDN'
    sdn_list_mapping = sanctionData[['uid', 'sanction_list_name']]
    sdn_list_mapping = sdn_list_mapping.drop_duplicates()
    sanctionData = sanctionData.drop('sanction_list_name', axis = 1)

    if input_type == 'Transaction Data':
        res = score_transaction(listFromClient, sanctionData, name_matching_method=name_matching_method, address_matching_method = address_matching_method, scoreTrsh=scoreTrsh, mini_batch=mini_batch, sep=sep)
    elif input_type == 'Physical Person - Individuals':
        res = score_physical(listFromClient, sanctionData, name_matching_method=name_matching_method, address_matching_method = address_matching_method, scoreTrsh=scoreTrsh, mini_batch=mini_batch, sep=sep)
    elif input_type == 'Moral Person - Companies':
        res = score_moral(listFromClient, sanctionData, name_matching_method=name_matching_method, address_matching_method = address_matching_method, scoreTrsh=scoreTrsh, mini_batch=mini_batch, sep=sep)
    elif input_type == 'Customers (mix PP MP)':
        res = score_customers(listFromClient, sanctionData, name_matching_method=name_matching_method, address_matching_method = address_matching_method, scoreTrsh=scoreTrsh, mini_batch=mini_batch, sep=sep)
    else:
        raise TypeError(f'"{input_type}" is an invalid keyword argument for score()')

    if input_type != 'Physical Person - Individuals':
        res[['DoB_client(multi_value)', 'DoB_sdn(multi_value)', 'matching_DoB_client', 'matching_DoB_sdn']] = np.NaN
    '''res = res.rename(columns={'uid': 'uid_sdn',
                              'Name': 'name_client',
                              'full_name_sdn': 'name_sdn',
                              'Address': 'address_client',
                              'address_y': 'address_sdn',
                              'city_x': 'city_client',
                              'city_y': 'city_sdn',
                              'Country': 'country_client',
                              'country_y': 'country_sdn',
                              'fromDoB': 'DoB_sdn'})'''


    res = res.merge(sdn_list_mapping, how='left', left_on='id_sdn', right_on='uid')

    res['review_result'] = np.NaN
    res['timestamp'] = np.NaN

    res = res[['id_client', 'id_sdn', 'sanction_list_name', 'Overall_Score', 
    'review_result', 'timestamp',
    #name
    'name_client(multi_value)',
       'name_sdn(multi_value)', 'matching_name_client', 'matching_name_sdn',
       'Name_Matching_%', 'score_Name', 
       #DoB
       'DoB_client(multi_value)',
       'DoB_sdn(multi_value)', 'matching_DoB_client', 'matching_DoB_sdn',
       'score_DoB', 
       # address
       'Address_client(multi_value)',
       'Address_sdn(multi_value)', 'matching_address_client',
       'matching_address_sdn', 'Country_client(multi_value)',
       'Country_sdn(multi_value)', 'matching_country_client',
       'matching_country_sdn', 'score_Address']]

    res['max_score'] = res.groupby('id_client')['Overall_Score'].transform('max')
    res['score_rank'] = res.groupby("id_client")["Overall_Score"].rank("dense", ascending=False)
    return res.sort_values(by=['max_score', 'id_client', 'score_rank'], ascending=[False,True,True])
