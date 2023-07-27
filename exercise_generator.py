## импорты

import spacy
import re
import pandas as pd
import pyinflect
from pyinflect import getAllInflections, getInflection
import numpy as np
import streamlit as st
from nltk.tokenize import sent_tokenize
import nltk
import random
from random import randint
#import gensim
#import gensim.downloader as api
#from multiprocessing import cpu_count
#from gensim.models.word2vec import Word2Vec
#from pattern3.en import lexeme, comparative, superlative
from io import StringIO, BytesIO
from gtts import gTTS
#import os
#from playsound import playsound
#import lemminflect

nltk.download('punkt')
nltk.download('wordnet')
random.seed(42)

###################### ФУНКЦИИ #########################

## проверка длины текста

def text_len_check(text):
    if len(text) <= 10000:
        if st.button('Показать текст', key='len_check'):
            st.write(text)
        text = text.replace('\n', ' ')
    else:
        st.write('Текст должен быть менее 10000 символов')

## выбор числа упражнений

def ex_num_slider(df):
    if len(df) == 0:
        st.write('')
        ex_num_option = 0
    else:
        with st.sidebar:
            ex_num_option = st.slider('Количество предложений:', 1,
                                      len(df),
                                      len(df)//4,
                                      key='slider')
    return ex_num_option

## счетчик правильных ответов

def option_check(option, corr_counter, answ):
    if option == '':
        st.write('')
    elif option.lower() == answ.lower():
        st.success('Правильный ответ', icon='✅')
        corr_counter += 1
    else:
        st.error('Попробуй еще раз', icon='❌')
    return corr_counter

## дырка в предложении c selectbox

def gap_func(df, ex_num_option):
    corr_counter = 0
    for i in range(ex_num_option):
        splitted = df.sent[i].split(df.answ[i])
        st.write('______'.join(splitted))
        option = st.selectbox('Выбери ответ: ', (df.answ_list[i]), key='gap'+str(i))
        corr_counter = option_check(option, corr_counter, df.answ[i])
    #if ex_num_option == 0:
    #    st.error('В тексте нет подходящих предложений.')
   #     ex_num_option = 1
        st.divider()

    return corr_counter

def answ_counter_func(corr_counter, ex_num_option):
    with st.sidebar:
        st.write('Правильные ответы:', corr_counter, 'из', ex_num_option)

#### глаголы

def verb_func(df):
    for i in range(len(df)):
        doc = nlp(df.sent[i])

        answ_list = []
        for token in doc:
            if token.pos_ == 'VERB':
               df.answ[i] = token.text
               answ_list = lexeme(df.answ[i])
               for form in answ_list:
                   if form.lower() == df.answ[i].lower():
                       answ_list.remove(form)
               answ_list = answ_list[:2]
               answ_list.append(df.answ[i].lower())
               random.shuffle(answ_list)
               answ_list.insert(0, '')
               df.answ_list[i] = answ_list
    df = df.dropna().reset_index(drop=True)
    ex_num_option = ex_num_slider(df)
    corr_counter = gap_func(df, ex_num_option)
    with st.sidebar:
        st.write('Правильные ответы:', corr_counter, 'из', ex_num_option)

def verb_func(df):
    for i in range(len(df)):
        doc = nlp(df.sent[i])

        answ_list = []
        for token in doc:
            if token.pos_ == 'VERB':
                df.answ[i] = token.text
                df.answ_list[i] = token._.inflect('VB')
    df = df.dropna().reset_index(drop=True)
    for i in range(len(df)):
        answ_list = []
        answ_list.append(list(getInflection(df.answ_list[i], tag='VBZ'))[0])
        answ_list.append(list(getInflection(df.answ_list[i], tag='VBG'))[0])
        answ_list.append(list(getInflection(df.answ_list[i], tag='VBD'))[0])
        answ_list.append(list(getInflection(df.answ_list[i], tag='VBP'))[0])
        answ_list.append('had ' + list(getInflection(df.answ_list[i], tag='VBN'))[0])

        for form in answ_list:
            if form.lower() == df.answ[i].lower():
                answ_list.remove(form)

        random.shuffle(answ_list)
        answ_list = answ_list[:3]
        answ_list.append(df.answ[i].lower())
        random.shuffle(answ_list)
        answ_list.insert(0, '')
        df.answ_list[i] = answ_list
    ex_num_option = ex_num_slider(df)
    corr_counter = gap_func(df, ex_num_option)

    answ_counter_func(corr_counter, ex_num_option)

## прилагательные


def adj_func(df):
    for i in range(len(df)):
        doc = nlp(df.sent[i])
        for token in doc:
            if token.pos_ == 'ADJ':
                df.answ[i] = token.text
                df.answ_list[i] = token._.inflect('JJ')

    df = df.dropna().reset_index(drop=True)

    for i in range(len(df)):
        answ_list = []
        answ_list.append(list(getInflection(df.answ_list[i], tag='JJ'))[0])
        answ_list.append(list(getInflection(df.answ_list[i], tag='JJR'))[0])
        answ_list.append('the ' + list(getInflection(df.answ_list[i], tag='JJS'))[0])

        for form in answ_list:
            if form.lower() == df.answ[i].lower():
                answ_list.remove(form)
        answ_list.append(df.answ[i].lower())
        random.shuffle(answ_list)
        answ_list.insert(0, '')
        df.answ_list[i] = answ_list

    ex_num_option = ex_num_slider(df)
    corr_counter = gap_func(df, ex_num_option)
    answ_counter_func(corr_counter, ex_num_option)

## порядок слов в предложении

def word_order_func(df):
    for i in range(len(df)):
        df.answ[i] = (df.sent[i].replace('\"', '').split())
        df.answ_list[i] = df.sent[i].replace('\"', '').split()
        if len(df.answ_list[i]) > 7:
            df.answ[i] = None
    df = df.dropna().reset_index(drop=True)
    ex_num_option = ex_num_slider(df)
    corr_counter = 0
    for i in range(ex_num_option):
        random.shuffle(df.answ_list[i])
        options = st.multiselect('Составь предложение:', df.answ_list[i])
        if options == []:
            st.write('')
        elif options == df.answ[i]:
            st.success('Верно!', icon='✅')
            corr_counter +=1
        else:
            st.error('Попробуй еще раз', icon='❌')
            if st.button('Показать ответ', key='show_answ'+str(i)):
                st.write(' '.join(df.answ[i]))
        st.divider()
    #if ex_num_option == 0:
    #    st.error('В тексте нет подходящих предложений.')
    #    ex_num_option = 1
    return corr_counter, ex_num_option


#### аудио ####

def audio_func(df):
    for i in range(len(df)):
        doc = nlp(df.sent[i])
        for token in doc:
            if (token.pos_ == 'NOUN') or (token.pos_ == 'ADJ'):
                df.answ[i] = token.text
                df.answ_list[i] = token.text

    df = df.dropna().reset_index(drop=True)
    ex_num_option = ex_num_slider(df)
    corr_counter = 0

    for i in range(ex_num_option):
        sound_file = BytesIO()
        audio = gTTS(text = df.sent[i], lang='en')
        audio.write_to_fp(sound_file)
        st.audio(sound_file)

        splitted = df.sent[i].split(df.answ[i])
        st.write('______'.join(splitted))
        option = st.text_input('Напиши свой ответ:', '', key='fill_the_gap'+str(i))
        corr_counter = option_check(option, corr_counter, df.answ[i])

    #if ex_num_option == 0:
    #        st.error('В тексте нет подходящих предложений.')
    #        ex_num_option = 1

        st.divider()

    #corr_counter = gap_func(df, ex_num_option)
    answ_counter_func(corr_counter, ex_num_option)

###################### ------- #########################


st.title('Генератор простых упражнений по английскому языку')
st.caption('Для продолжения выбери текстовый файл или введи свой текст и нажми Enter')
with st.sidebar:
    st.subheader('Параметры:')

## загрузка текста

text = ''
file = st.file_uploader('Выбрать файл(.txt):')
if file is not None:
    stringio = StringIO(file.getvalue().decode("utf-8"))
    text = stringio.read()
    text_len_check(text)
else:
    text = st.text_input('Ввести свой текст:', '', key='text_input')
    text_len_check(text)



################ создание датафрейма #########################

sentences = sent_tokenize(text)
df = []
for sent in sentences:
    df.append(sent)

df = pd.DataFrame(df).rename(columns={0: 'sent'})

ex_type_list = ['verb_time', 'adjective_form', 'word_order', 'audio']  # список имен упражнений

df['answ'] = np.nan
df['answ_list'] = np.nan
#df['ex_type'] = np.nan


#for i in range(len(df)):
#    df['ex_type'][i] = ex_type_list[randint(0, len(ex_type_list)-1)]


## выбор упражнения
with st.sidebar:
    ex_option = st.radio('Тип упражения:',
        ('Времена глаголов', 'Форма прилагательных',
         'Порядок слов в предложении', 'Аудио'), key='ex_type')

st.write('Упражнение:  ', ex_option)
nlp = spacy.load('en_core_web_sm')


## выбор времена глаголов

if ex_option == 'Времена глаголов':
    st.caption('Заполни пробел в предожении, выбрав вариант из выпадающего списка')

    #df = df.query('ex_type == "verb_time"').reset_index(drop=True)

    verb_func(df)


## выбор форма прилагательных

if ex_option == 'Форма прилагательных':
    st.caption('Заполни пробел в предожении, выбрав вариант из выпадающего списка')

    #df = df.query('ex_type == "adjective_form"').reset_index(drop=True)
    adj_func(df)


    ## выбор порядок слов

if ex_option == 'Порядок слов в предложении':
    #df = df.query('ex_type == "word_order"').reset_index(drop=True)
    st.caption('Составь предложение в правильном порядке, выбрав варианты из выпадающего списка')


    corr_counter, ex_num_option = word_order_func(df)
    answ_counter_func(corr_counter, ex_num_option)

    ## выбор аудио


if ex_option == 'Аудио':
    #df = df.query('ex_type == "audio"').reset_index(drop=True)
    st.caption('Прослушай предложение и заполни пробел, написав слово в ячейке')

    audio_func(df)














































