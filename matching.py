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


i =  0
def compare_names_with_progress(full_name_client, full_name_sdn, name_matching_method, total, start):
    global i
    i += 1
    if i % 5000 == 0:
        s = i/total
        sec_passed = time.perf_counter() - start
        sec_remaining = (sec_passed / s) * (1 - s)
        print(f"{i}/{total} ({s * 100:.1f}%)  Time passed={seconds_to_str(sec_passed)}  ; Time remaining={seconds_to_str(sec_remaining)} ")
    return compare_names(full_name_client, full_name_sdn, name_matching_method=name_matching_method)

def score_transaction(listFromClient, sanctionData, name_matching_method, address_matching_method):
    res = pd.merge(listFromClient, sanctionData, how='cross')
    
    #Scoring names
    start = time.perf_counter()
    res['Name_Matching_%'], res['score_Name'] = zip(*res.apply(lambda x: compare_names_with_progress(x["full_name_client"], x["full_name_sdn"], name_matching_method=name_matching_method, total = res.shape[0], start = start), axis=1))

    #Scoring addresses
    if address_matching_method == 'exact_match':
        res['scored'] = False
        fields = ['score_Address', 'scored']
        #Exact math of address, city and country
        res.loc[(res['address_x'] == res['address_y']) &
                #(res['city_x'] == res['city_y']) &
                (res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 500, True
        #Match of address and city only
        res.loc[(res['address_x'] == res['address_y']) &
                #(res['city_x'] == res['city_y']) &
                ~res['scored'], fields] = 100, True
        #Match of address and country only
        res.loc[(res['address_x'] == res['address_y']) &
                (res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 400, True
        #Match of city and country only
        res.loc[#(res['city_x'] == res['city_y']) &
                (res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 300, True
        #Match of address only
        res.loc[(res['address_x'] == res['address_y']) &
                ~res['scored'], fields] = 0, True
        #Match of city only
        #res.loc[(res['city_x'] == res['city_y']) &
                #~res['scored'], fields] = -100, True
        #Match of country only
        res.loc[(res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 200, True
        #No data
        res.loc[(res['country_x'].isna()) |
                (res['country_y'].isna()) &
                ~res['scored'], fields] = 0, True
        #No match
        res.loc[~res['scored'], fields] = -200, True
    #elif address_matching_method == 'fuzzy_match':
        #res["score_Address"] = res.apply(lambda x: compare_addresses(x, address_matching_method=address_matching_method), axis=1)
    else:
        raise TypeError(f'"{address_matching_method}" is an invalid keyword argument for score()')
    
    res['Overall_Score'] = res['score_Name'] + res['score_Address']

    res = res.drop_duplicates()

    res[['DoB', 'fromDoB', 'score_DoB']] = np.NaN
    
    return res


def score_physical(listFromClient, sanctionData, name_matching_method, address_matching_method):
    #listFromClient['DateOfBirth'] = listFromClient['DateOfBirth'].str.replace('circa ', '')
    sanctionData['dateOfBirth'] = sanctionData['dateOfBirth'].str.replace('circa', 'to')
    sanctionData['isRange'] = sanctionData['dateOfBirth'].str.contains('to').fillna(False)
    sanctionData[['fromDoB', 'toDoB']] = sanctionData['dateOfBirth'].str.split('to ', n=2, expand=True)
    sanctionData = sanctionData.drop(columns=['dateOfBirth'])
    
    sanctionData['fromDoB'] = pd.to_datetime(sanctionData['fromDoB'], infer_datetime_format=True)
    sanctionData['toDoB'] = pd.to_datetime(sanctionData['toDoB'], infer_datetime_format=True)
    listFromClient['DoB'] = pd.to_datetime(listFromClient['DoB'], infer_datetime_format=True)

    res = pd.merge(listFromClient, sanctionData, how='cross')

    #Scoring names
    start = time.perf_counter()
    res['Name_Matching_%'], res['score_Name'] = zip(*res.apply(lambda x: compare_names_with_progress(x["full_name_client"], x["full_name_sdn"], name_matching_method=name_matching_method, total = res.shape[0], start = start), axis=1))

    #Scoring dates
    res['l_year']   = res['DoB'].dt.year
    res['l_month']  = res['DoB'].dt.month
    res['l_day']    = res['DoB'].dt.day
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
    res.loc[(res['DoB'] == res['fromDoB']) &
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
    res.loc[(np.abs(res['fromDoB'] - res['DoB']) /  np.timedelta64(1, 'Y') < 2) &
            ~res['isRange'] &
            ~res['scored'], fields] = 100, True
    ## More than 2 years difference in DoB between bank and World-Check’ information
    res.loc[(np.abs(res['fromDoB'] - res['DoB']) /  np.timedelta64(1, 'Y') >= 2) &
            ~res['isRange'] &
            ~res['scored'], fields] = -200, True
    ## Date of Birth not recorded on Bank or World-Check’ list
    res.loc[~res['isRange'] & 
            ~res['scored'], fields] = 0, True 
    # === Case when DoB in sanction data is a range ===
    ## Date is before ('circa' case)
    res.loc[(res['DoB'] <= res['toDoB']) &
            (res['fromDoB'] is None) &
            ~res['scored'], fields] = 100, True
    ## Date is in range
    res.loc[(res['DoB'] <= res['toDoB']) &
            (res['fromDoB'] <= res['DoB']) &
            ~res['scored'], fields] = 150, True
    ## Date is not in range
    res.loc[~res['scored'], fields] = 0, True

    #Scoring addresses
    if address_matching_method == 'exact_match':
        res['scored'] = False
        fields = ['score_Address', 'scored']
        #Exact math of address, city and country
        res.loc[(res['address_x'] == res['address_y']) &
                #(res['city_x'] == res['city_y']) &
                (res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 500, True
        #Match of address and city only
        res.loc[(res['address_x'] == res['address_y']) &
                #(res['city_x'] == res['city_y']) &
                ~res['scored'], fields] = 100, True
        #Match of address and country only
        res.loc[(res['address_x'] == res['address_y']) &
                (res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 400, True
        #Match of city and country only
        res.loc[#(res['city_x'] == res['city_y']) &
                (res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 300, True
        #Match of address only
        res.loc[(res['address_x'] == res['address_y']) &
                ~res['scored'], fields] = 0, True
        #Match of city only
        #res.loc[(res['city_x'] == res['city_y']) &
                #~res['scored'], fields] = -100, True
        #Match of country only
        res.loc[(res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 200, True
        #No data
        res.loc[(res['country_x'].isna()) |
                (res['country_y'].isna()) &
                ~res['scored'], fields] = 0, True
        #No match
        res.loc[~res['scored'], fields] = -200, True
    #elif address_matching_method == 'fuzzy_match':
        #res["score_Address"] = res.apply(lambda x: compare_addresses(x, address_matching_method=address_matching_method), axis=1)
    else:
        raise TypeError(f'"{address_matching_method}" is an invalid keyword argument for score()')
    
    res['Overall_Score'] = res['score_Name'] + res['score_DoB'] + res['score_Address']

    res = res.drop_duplicates()
    
    return res

def score_moral(listFromClient, sanctionData, name_matching_method, address_matching_method):
    res = pd.merge(listFromClient, sanctionData, how='cross')

    #Scoring names
    start = time.perf_counter()
    res['Name_Matching_%'], res['score_Name'] = zip(*res.apply(lambda x: compare_names_with_progress(x["full_name_client"], x["full_name_sdn"], name_matching_method=name_matching_method, total = res.shape[0], start = start), axis=1))

        #Scoring addresses
    if address_matching_method == 'exact_match':
        res['scored'] = False
        fields = ['score_Address', 'scored']
        #Exact math of address, city and country
        res.loc[(res['address_x'] == res['address_y']) &
                #(res['city_x'] == res['city_y']) &
                (res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 500, True
        #Match of address and city only
        res.loc[(res['address_x'] == res['address_y']) &
                #(res['city_x'] == res['city_y']) &
                ~res['scored'], fields] = 100, True
        #Match of address and country only
        res.loc[(res['address_x'] == res['address_y']) &
                (res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 400, True
        #Match of city and country only
        res.loc[#(res['city_x'] == res['city_y']) &
                (res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 300, True
        #Match of address only
        res.loc[(res['address_x'] == res['address_y']) &
                ~res['scored'], fields] = 0, True
        #Match of city only
        #res.loc[(res['city_x'] == res['city_y']) &
                #~res['scored'], fields] = -100, True
        #Match of country only
        res.loc[(res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 200, True
        #No data
        res.loc[(res['country_x'].isna()) |
                (res['country_y'].isna()) &
                ~res['scored'], fields] = 0, True
        #No match
        res.loc[~res['scored'], fields] = -200, True
    #elif address_matching_method == 'fuzzy_match':
        #res["score_Address"] = res.apply(lambda x: compare_addresses(x, address_matching_method=address_matching_method), axis=1)
    else:
        raise TypeError(f'"{address_matching_method}" is an invalid keyword argument for score()')
    
    res['Overall_Score'] = res['score_Name'] + res['score_Address']

    res = res.drop_duplicates()

    res[['DoB', 'fromDoB', 'score_DoB']] = np.NaN
    
    return res    #Scoring names
    res = pd.merge(listFromClient, sanctionData, how='cross')

    res["score_Name"] = res.apply(lambda x: compare_names_with_progress(x, name_matching_method, res.shape[0]), axis=1)

        #Scoring addresses
    if address_matching_method == 'exact_match':
        res['scored'] = False
        fields = ['score_Address', 'scored']
        #Exact math of address, city and country
        res.loc[(res['address_x'] == res['address_y']) &
                #(res['city_x'] == res['city_y']) &
                (res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 500, True
        #Match of address and city only
        res.loc[(res['address_x'] == res['address_y']) &
                #(res['city_x'] == res['city_y']) &
                ~res['scored'], fields] = 100, True
        #Match of address and country only
        res.loc[(res['address_x'] == res['address_y']) &
                (res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 400, True
        #Match of city and country only
        res.loc[#(res['city_x'] == res['city_y']) &
                (res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 300, True
        #Match of address only
        res.loc[(res['address_x'] == res['address_y']) &
                ~res['scored'], fields] = 0, True
        #Match of city only
        #res.loc[(res['city_x'] == res['city_y']) &
                #~res['scored'], fields] = -100, True
        #Match of country only
        res.loc[(res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 200, True
        #No data
        res.loc[(res['country_x'].isna()) |
                (res['country_y'].isna()) &
                ~res['scored'], fields] = 0, True
        #No match
        res.loc[~res['scored'], fields] = -200, True
    #elif address_matching_method == 'fuzzy_match':
        #res["score_Address"] = res.apply(lambda x: compare_addresses(x, address_matching_method=address_matching_method), axis=1)
    else:
        raise TypeError(f'"{address_matching_method}" is an invalid keyword argument for score()')
    
    res['Overall_Score'] = res['score_Name'] + res['score_Address']

    res = res[['id_client', 'recordID_sdn', 'uid', 'full_name_client', 'full_name_sdn', 'address_x', 'address_y', 'country_x',
               'country_y', 'score_Name', 'score_Address', 'Overall_Score']]

    res = res.drop_duplicates()

    res = res.rename(columns={'uid': 'uid_sdn',
                              'address_x': 'address_client',
                              'address_y': 'address_sdn',
                              'country_x': 'country_client',
                              'country_y': 'country_sdn'})
    
    return res

def score_customers(listFromClient, sanctionData, name_matching_method='bleu', address_mathing='exact_match', address_matching_method='bleu'):
    
    res = pd.merge(listFromClient, sanctionData, how='cross')

    #Scoring names
    start = time.perf_counter()
    res['Name_Matching_%'], res['score_Name'] = zip(*res.apply(lambda x: compare_names_with_progress(x["full_name_client"], x["full_name_sdn"], name_matching_method=name_matching_method, total = res.shape[0], start = start), axis=1))

    #Scoring addresses
    if address_matching_method == 'exact_match':
        res['scored'] = False
        fields = ['score_Address', 'scored']
        #Exact math of address, city and country
        res.loc[(res['address_x'] == res['address_y']) &
                #(res['city_x'] == res['city_y']) &
                (res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 500, True
        #Match of address and city only
        res.loc[(res['address_x'] == res['address_y']) &
                #(res['city_x'] == res['city_y']) &
                ~res['scored'], fields] = 100, True
        #Match of address and country only
        res.loc[(res['address_x'] == res['address_y']) &
                (res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 400, True
        #Match of city and country only
        res.loc[#(res['city_x'] == res['city_y']) &
                (res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 300, True
        #Match of address only
        res.loc[(res['address_x'] == res['address_y']) &
                ~res['scored'], fields] = 0, True
        #Match of city only
        #res.loc[(res['city_x'] == res['city_y']) &
                #~res['scored'], fields] = -100, True
        #Match of country only
        res.loc[(res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 200, True
        #No data
        res.loc[(res['country_x'].isna()) |
                (res['country_y'].isna()) &
                ~res['scored'], fields] = 0, True
        #No match
        res.loc[~res['scored'], fields] = -200, True
    #elif address_matching_method == 'fuzzy_match':
        #res["score_Address"] = res.apply(lambda x: compare_addresses(x, address_matching_method=address_matching_method), axis=1)
    else:
        raise TypeError(f'"{address_matching_method}" is an invalid keyword argument for score()')
    
    res['Overall_Score'] = res['score_Name'] + res['score_Address']

    res = res.drop_duplicates()

    res[['DoB', 'fromDoB', 'score_DoB']] = np.NaN

    return res    #Scoring names
    res = pd.merge(listFromClient, sanctionData, how='cross')

    res["score_Name"] = res.apply(lambda x: compare_names_with_progress(x, name_matching_method, res.shape[0]), axis=1)

    #Scoring addresses
    if address_matching_method == 'exact_match':
        res['scored'] = False
        fields = ['score_Address', 'scored']
        #Exact math of address, city and country
        res.loc[(res['address_x'] == res['address_y']) &
                #(res['city_x'] == res['city_y']) &
                (res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 500, True
        #Match of address and city only
        res.loc[(res['address_x'] == res['address_y']) &
                #(res['city_x'] == res['city_y']) &
                ~res['scored'], fields] = 100, True
        #Match of address and country only
        res.loc[(res['address_x'] == res['address_y']) &
                (res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 400, True
        #Match of city and country only
        res.loc[#(res['city_x'] == res['city_y']) &
                (res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 300, True
        #Match of address only
        res.loc[(res['address_x'] == res['address_y']) &
                ~res['scored'], fields] = 0, True
        #Match of city only
        #res.loc[(res['city_x'] == res['city_y']) &
                #~res['scored'], fields] = -100, True
        #Match of country only
        res.loc[(res['country_x'] == res['country_y']) &
                ~res['scored'], fields] = 200, True
        #No data
        res.loc[(res['country_x'].isna()) |
                (res['country_y'].isna()) &
                ~res['scored'], fields] = 0, True
        #No match
        res.loc[~res['scored'], fields] = -200, True
    #elif address_matching_method == 'fuzzy_match':
        #res["score_Address"] = res.apply(lambda x: compare_addresses(x, address_matching_method=address_matching_method), axis=1)
    else:
        raise TypeError(f'"{address_matching_method}" is an invalid keyword argument for score()')
    
    res['Overall_Score'] = res['score_Name'] + res['score_Address']

    res = res[['id_client', 'recordID_sdn', 'uid', 'full_name_client', 'full_name_sdn', 'address_x', 'address_y', 'country_x',
               'country_y', 'score_Name', 'score_Address', 'Overall_Score']]

    res = res.drop_duplicates()

    res = res.rename(columns={'uid': 'uid_sdn',
                              'address_x': 'address_client',
                              'address_y': 'address_sdn',
                              'country_x': 'country_client',
                              'country_y': 'country_sdn'})
    
    return res

def score(listFromClient, sanctionData, input_type, name_matching_method='bleu', address_matching_method = 'exact_match', scoreTrsh = 800, sep = '¦'):
    listFromClient = listFromClient.assign(full_name_client=listFromClient['Name'].str.split(sep)).explode('full_name_client')
    listFromClient = listFromClient.assign(address=listFromClient['Address'].str.split(sep)).explode('address')
    listFromClient = listFromClient.assign(country=listFromClient['Country'].str.split(sep)).explode('country')
    try:
        listFromClient = listFromClient.assign(DoB=listFromClient.DateOfBirth.str.split(sep)).explode('DoB')
    except:
        pass
    
    if input_type == 'Transaction Data':
        res = score_transaction(listFromClient, sanctionData, name_matching_method=name_matching_method, address_matching_method = address_matching_method)
    elif input_type == 'Physical Person - Individuals':
        res = score_physical(listFromClient, sanctionData, name_matching_method=name_matching_method, address_matching_method = address_matching_method)
    elif input_type == 'Moral Person - Companies':
        res = score_moral(listFromClient, sanctionData, name_matching_method=name_matching_method, address_matching_method = address_matching_method)
    elif input_type == 'Customers (mix PP MP)':
        res = score_customers(listFromClient, sanctionData, name_matching_method=name_matching_method, address_matching_method = address_matching_method)
    else:
        raise TypeError(f'"{input_type}" is an invalid keyword argument for score()')

    if 'DateOfBirth' not in res.columns:
        res['DateOfBirth'] = np.nan
    res = res.rename(columns={'uid': 'uid_sdn',
                              'Name': 'name_client',
                              'full_name_sdn': 'name_sdn',
                              'Address': 'address_client',
                              'address_y': 'address_sdn',
                              'city_x': 'city_client',
                              'city_y': 'city_sdn',
                              'Country': 'country_client',
                              'country_y': 'country_sdn',
                              'DateOfBirth': 'DoB_client',
                              'fromDoB': 'DoB_sdn'})

    res = res[['UniqueID', 'recordID_sdn', 'uid_sdn', 'name_client', 'name_sdn', 'DoB_client', 'DoB_sdn', 'address_client', 'address_sdn', 'country_client',
               'country_sdn', 'Name_Matching_%', 'score_Name', 'score_DoB', 'score_Address', 'Overall_Score']]
        
    return res.query(f'Overall_Score >= {scoreTrsh}').sort_values('Overall_Score', ascending=False)