## импорты

import nltk
import numpy as np
import pandas as pd
import pyinflect
import random
import re
import spacy
import streamlit as st

from io import StringIO, BytesIO
from gtts import gTTS
from nltk.tokenize import sent_tokenize
from pyinflect import getAllInflections, getInflection

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
                                      len(df) // 3,            # показываем по умолчанию треть упражнений                     
                                      key='slider')
    return ex_num_option

## счетчик правильных ответов

def option_check(option, corr_counter, answ, i):
    if option == '':
        st.write('')
    elif option.lower() == answ.lower():
        st.success('Правильный ответ', icon='✅')
        corr_counter += 1
    else:
        st.error('Попробуйте еще раз', icon='❌')
        if st.button('Показать ответ', key='show_answ'+str(i)):
            st.write(answ)
    return corr_counter

## info 

def info_func(ex_num_option):
    if ex_num_option == 0:
        st.info('🤔 Похоже, тут ничего нет. Возможные причины: выбрано нулевое число упражнений, или в тексте нет подходящих предложений. Попробуйте загрузить другой текст😅')

## дырка в предложении c selectbox и счетчиком

def gap_func(df, ex_num_option):

    corr_counter = 0
    info_func(ex_num_option)

    for i in range(ex_num_option):
        
        splitted = df.sent[i].split(df.answ[i])    # делаем сплит по ответу, дополнить позже
        st.write('______'.join(splitted))
        option = st.selectbox('Выбери ответ: ', (df.answ_list[i]), key='gap'+str(i))
        corr_counter = option_check(option, corr_counter, df.answ[i], i)
        
        st.divider()

    return corr_counter         

## вывод кол-ва верных ответов

def answ_counter_func(corr_counter, ex_num_option):
    with st.sidebar:
        st.write('Правильные ответы:', corr_counter, 'из', ex_num_option)

## упражнение на время глаголов

def verb_func(df):
    for i in range(len(df)):
        doc = nlp(df.sent[i])

        for token in doc:
            if token.pos_ == 'VERB':
                df.answ[i] = token.text
                df.answ_list[i] = token._.inflect('VB')  

    df = df.dropna().reset_index(drop=True)
    answ_list = set()
    forms_list = ['VBZ', 'VBG', 'VBD', 'VBP']

    for i in range(len(df)):

        answ_list = {list(getInflection(df.answ_list[i].lower(), tag=form))[0] for form in forms_list}
        answ_list.add('had ' + list(getInflection(df.answ_list[i].lower(), tag='VBN'))[0])
        df.answ_list[i] = list(answ_list)
        df.answ_list[i].insert(0, '')

    ex_num_option = ex_num_slider(df)
    corr_counter = gap_func(df, ex_num_option)
    answ_counter_func(corr_counter, ex_num_option)


## упражнение на форму прилагательных


def adj_func(df):
    for i in range(len(df)):
        doc = nlp(df.sent[i])
        for token in doc:
            if token.pos_ == 'ADJ':
                df.answ[i] = token.text
                df.answ_list[i] = token._.inflect('JJ')

    df = df.dropna().reset_index(drop=True)

    for i in range(len(df)):
        answ_list = set()
        answ_list.add(list(getInflection(df.answ_list[i], tag='JJ'))[0].lower())
        answ_list.add(list(getInflection(df.answ_list[i], tag='JJR'))[0].lower())
        answ_list.add('the ' + list(getInflection(df.answ_list[i], tag='JJS'))[0].lower())
        answ_list.add(df.answ[i].lower())

        df.answ_list[i] = list(answ_list)
        df.answ_list[i].insert(0, '')

    ex_num_option = ex_num_slider(df)
    corr_counter = gap_func(df, ex_num_option)
    answ_counter_func(corr_counter, ex_num_option)

## порядок слов в предложении

def word_order_func(df):
    for i in range(len(df)):

        df.answ[i] = (df.sent[i].replace('\"', '').split())
        df.answ_list[i] = df.sent[i].replace('\"', '').split()

        if len(df.answ_list[i]) > 7:   # оставляем короткие предложения для удобства и красоты           
            df.answ[i] = None

    df = df.dropna().reset_index(drop=True)
    ex_num_option = ex_num_slider(df)
    corr_counter = 0
    info_func(ex_num_option)

    for i in range(ex_num_option):

        random.shuffle(df.answ_list[i])
        options = st.multiselect('Составьте предложение:', df.answ_list[i]) # выбор слов в мультиселект
        
        if options == []:                                                 # и счетчик правильных ответов
            st.write('')

        elif options == df.answ[i]:
            st.success('Верно!', icon='✅')
            corr_counter +=1

        else:
            st.error('Попробуйте еще раз', icon='❌')
            if st.button('Показать ответ', key='show_answ'+str(i)):
                st.write(' '.join(df.answ[i]))

        st.divider()

    return corr_counter, ex_num_option


## упражение аудио 

def audio_func(df):
    for i in range(len(df)):

        doc = nlp(df.sent[i])
        for token in doc:

            if (token.pos_ == 'NOUN') or (token.pos_ == 'ADJ'): # выбираем прилагательные или существительные
                df.answ[i] = token.text
                df.answ_list[i] = token.text

    df = df.dropna().reset_index(drop=True)
    ex_num_option = ex_num_slider(df)
    corr_counter = 0
    info_func(ex_num_option)

    for i in range(ex_num_option):

        sound_file = BytesIO()
        audio = gTTS(text = df.sent[i], lang='en')
        audio.write_to_fp(sound_file)
        st.audio(sound_file)

        splitted = df.sent[i].split(df.answ[i])
        st.write('______'.join(splitted))
        option = st.text_input('Напишите свой ответ:', '', key='fill_the_gap'+str(i))
        corr_counter = option_check(option, corr_counter, df.answ[i], i)

        st.divider()

    answ_counter_func(corr_counter, ex_num_option)

###################### ВЫПОЛНЕНИЕ #########################

st.set_page_config(page_title='SEGen', page_icon=':abc:')

st.title('Генератор простых упражнений по английскому языку')
st.caption('Для продолжения выберите текстовый файл или введи свой текст и нажми Enter')

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
    text = st.text_input('Ввести свой текст:', '', max_chars=10000, key='text_input', help='Лучше загружать текст из 10 предложений и более')
    if text is not None:
        text_len_check(text)

## создание датафрейма 

sentences = sent_tokenize(text)     # разбиваем текст на предложения
df = []

for sent in sentences:
    df.append(sent)

df = pd.DataFrame(df).rename(columns={0: 'sent'})
df['answ'] = np.nan         
df['answ_list'] = np.nan        # заполняем столбцы пустыми значениями

nlp = spacy.load('en_core_web_sm')


## выбор упражнения


with st.sidebar:
    ex_option = st.radio('Тип упражения:',
        ('Времена глаголов', 'Форма прилагательных',
         'Порядок слов в предложении', 'Аудио'), key='ex_type')

st.write('Упражнение:  ', ex_option)


## выбор времена глаголов

if ex_option == 'Времена глаголов':
    st.caption('Заполните пробел в предложении, выбрав вариант из выпадающего списка')

    verb_func(df)


## выбор форма прилагательных

if ex_option == 'Форма прилагательных':
    st.caption('Заполните пробел в предложении, выбрав вариант из выпадающего списка')

    adj_func(df)


## выбор порядок слов

if ex_option == 'Порядок слов в предложении':

    st.caption('Составьте предложение в правильном порядке, выбрав варианты из выпадающего списка')

    corr_counter, ex_num_option = word_order_func(df)
    answ_counter_func(corr_counter, ex_num_option)


## выбор аудио

if ex_option == 'Аудио':

    st.caption('Прослушайте предложение и заполните пробел, написав слово в ячейке')

    audio_func(df)














































