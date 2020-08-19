#script takes restaurant menus computes cosine similarity and then from user specified 
#restaurant returns "n" restaurants with the highest similarity
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #using cpu not gpu

import re
import numpy as np
import pandas as pd
import time
import datetime as dt
import random
import sqlite3

from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial

import tensorflow_hub as hub

import tokenization
import tensorflow as tf

conn = sqlite3.connect('ot.db')
module_url = "https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/2"
random_state = 12345
    
def urlClean(x):
    if 'https://www.opentable.comhttps://' in x:
        x = x.replace('https://www.opentable.comhttps://','https://')
    return x

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def cleanRestaurant(x):
    try:
        x = re.split(r"\s{2,}", x)[0]
        if x.endswith(' New'):
            x = x[:-4]
        return x.strip()
    except:
        return x
    
def getMostSimilar(name, s, restaurantMenus, n = 5, t = True):
    
    c = restaurantMenus['cuisine'][ restaurantMenus['restaurant'] == name ].unique()[0]
    ID = restaurantMenus['url'][ restaurantMenus['restaurant'] == name ].unique()[0]
    y = s.loc[ID]
    y.sort_values(ascending = False, inplace = True)
    
    if type(c) != type(None):
        sameCategory = restaurantMenus['url'][ restaurantMenus['cuisine'] == c ].tolist()
        y = y[ (y.index.isin(sameCategory)) ]
    y = y[ y > y.quantile(.9) ]
    z = restaurantMenus[ (restaurantMenus['url'].isin(y.index)) & (restaurantMenus['url'] != ID) ].sort_values('rating',ascending = False)
    
    return z['restaurant'].head(n)
    
if __name__ == '__main__':
    
    dinner = pd.read_sql_query("SELECT * FROM dinner", conn)
    brunch = pd.read_sql_query("SELECT * FROM brunch", conn)
    
    T = list(set(dinner['reservationDate'].tolist()))
    T.sort()
    TKey = dict(zip(T,[ pd.to_datetime(x) for x in T]))
    dinner['reservationDate'] = dinner['reservationDate'].map(TKey)
    
    T = list(set(brunch['reservationDate'].tolist()))
    T.sort()
    TKey = dict(zip(T,[ pd.to_datetime(x) for x in T]))
    brunch['reservationDate'] = brunch['reservationDate'].map(TKey)
    
    dinner.sort_values('reservationDate', ascending = True, inplace = True)
    brunch.sort_values('reservationDate', ascending = True, inplace = True)
    
    dinner['link'] = dinner['link'].apply(urlClean)
    brunch['link'] = brunch['link'].apply(urlClean)
    
    meals = pd.concat((dinner, brunch[dinner.columns.tolist()]))
    meals.reset_index(drop = True, inplace = True)
    meals.rename(columns = {'link':'url'}, inplace = True)
    meals.drop_duplicates('url', keep = 'first', inplace = True)
    
    allMenus = pd.read_sql_query("SELECT * FROM restaurant_menus", conn)
    allMenus.loc[ allMenus['desc'].isnull(),'desc'] = ''
    allMenus.loc[ allMenus['item'].isnull(),'item'] = ''
    allMenus['text'] = allMenus['item'] + '___' + allMenus['desc'] + '___'
    allMenus['text'] = allMenus['text'].str.replace('______','')
    allMenus['text'] = allMenus['text'].str.replace('___',': ')
    replacement = allMenus.loc[ allMenus['text'].str.endswith(': '), 'text' ].apply( lambda x: x[:-2] + '.')
    allMenus.loc[ allMenus['text'].str.endswith(': '), 'text' ] = replacement
    replacement = allMenus.loc[ allMenus['text'].str.startswith(': '), 'text' ].apply( lambda x: x[2:])
    allMenus.loc[ allMenus['text'].str.startswith(': '), 'text' ] = replacement
    allMenus = allMenus[ ~allMenus['text'].isnull() ]
    allMenus['text'] = allMenus['text'].str.lower()
    
    restaurantMenus = allMenus.groupby(['url'])['text'].apply( lambda x: ' '.join(x) )
    restaurantMenus = pd.DataFrame({'url':restaurantMenus.index,'text':restaurantMenus.values})
    restaurantMenus = restaurantMenus.merge(meals, how = 'left', on = ['url'])
    restaurantMenus.loc[ restaurantMenus['cuisine'] == 'Contemporary American', 'cuisine' ] = 'American'
    restaurantMenus['restaurant'] = restaurantMenus['restaurant'].apply(cleanRestaurant)
    restaurantMenus.loc[ restaurantMenus['restaurant'].isnull(), 'restaurant' ] = restaurantMenus.loc[ restaurantMenus['restaurant'].isnull(), 'url' ]
    restaurantMenus.drop_duplicates('restaurant',keep = 'first', inplace = True)
    
    bert_layer = hub.KerasLayer(module_url, trainable=True)
    
    max_len = 300
    
    vocabFile = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    doLowerCase = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocabFile, doLowerCase)
    
    trainInput = bert_encode(restaurantMenus['text'].values, tokenizer, max_len=max_len)
    
    X = trainInput[0]
    s = cosine_similarity(X)
    s = pd.DataFrame(s, index = restaurantMenus['url'], columns = restaurantMenus['url'])
    
    name = 'Highlands' #example
    
    print(getMostSimilar(name, s, restaurantMenus, n = 10, t = True))
    
