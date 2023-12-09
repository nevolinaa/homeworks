from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def get_images():
    correlation = Image.open('hw3/pictures/correlation.png')
    credit_fstpayment = Image.open('hw3/pictures/credit_fstpayment.png')
    dependants_child = Image.open('hw3/pictures/dependants_child.png')
    age_target = Image.open('hw3/pictures/age_target.png')
    gender_target = Image.open('hw3/pictures/gender_target.png')
    education_target = Image.open('hw3/pictures/education_target.png')
    income_target = Image.open('hw3/pictures/income_target.png')
    
    return correlation, credit_fstpayment, dependants_child, age_target, gender_target, education_target, income_target

def plot_features():
    df = st.cache_data(pd.read_csv)("hw3/data/df.csv") 
    correlation, credit_fstpayment, dependants_child, age_target, gender_target, education_target, income_target = get_images()

    
    st.subheader('Графики распределения: бинарные переменные')
    bin_cols = {
                'TARGET': ['отклика не было', 'отклик был'],
                'GENDER': ['Женщина', 'Мужчина'],
                'FL_PRESENCE_FL': ['Квартира есть', 'Квартиры нет'],
                'CLOSED_FL': ['Кредит закрыт', 'Кредит не закрыт'],
                'SOCSTATUS_WORK_FL': ['не работает', 'работает'],
                'SOCSTATUS_PENS_FL': ['не пенсионер', 'пенсионер'],
               }
 
    feature = st.selectbox('Выберите признак:',
                        list(bin_cols.keys()))
    df_feature = df.groupby(by=feature).AGREEMENT_RK.count().reset_index()
    sizes = df_feature.AGREEMENT_RK
    
    fig_pie, ax_pie = plt.subplots(figsize=(3, 3))

    ax_pie.pie(sizes, labels=bin_cols.get(feature),
               colors=['#982d80', '#f8765c'])
    plt.title(f'Распределение переменной {feature}')
    st.pyplot(fig_pie)
    
    st.subheader('Выводы')
    st.write('В основном в выборке у клиентов не было отклика на маркетинговое предложение. Выборка несбалансирована')
    st.write('В выборке больше мужчин')
    st.write('В выборке больше людей, которые не имеют собственную квартиру')
    st.write('Большая часть кредитов в выборке закрыта')
    st.write('В основном люди в выборке работающие')
    st.write('В основном люди в выборке не пенсионеры')

    
    st.subheader('Графики распределения: категориальные переменные')
    cat_cols = ['OWN_AUTO', 'EDUCATION', 'MARITAL_STATUS', 'FAMILY_INCOME', 
               'CHILD_TOTAL', 'TERM', 'GEN_INDUSTRY', 'GEN_TITLE', 'JOB_DIR']
    feature = st.selectbox('Выберите признак:', cat_cols)
    df_feature = df.groupby(by=feature).AGREEMENT_RK.count().reset_index()
    sizes = df_feature.AGREEMENT_RK
    fig, ax = plt.subplots()
    ax.bar(df_feature[feature], sizes, color='#982d80')
    plt.title(f'Распределение значений по {feature}')
    plt.xticks(rotation=90)
    st.pyplot(fig)
    
    st.subheader('Выводы')
    st.write('В основном люди в выборке не имеют машины')
    st.write('В основном люди в выборке с средним и средне-специальным образованием')
    st.write('В основном люди в выборке состоят в браке')
    st.write('В основном люди в выборке с доходами от 10 до 50 тысяч на семью')
    st.write('В выборке в основном представлены семьи с 1-2 детьми')
    st.write('Видим, что есть несколько крупных групп: с кредитам на 3,6,10 и 12 лет')
    st.write('В основном в выборке представлены работающие в торговле люди')
    st.write('В основном в выборке представлены специалисты и рабочие')

    
    st.subheader('Графики распределения: вещественные переменные')
    vesh_cols = ['AGE', 'PERSONAL_INCOME', 'CREDIT', 'FST_PAYMENT']
    feature = st.selectbox('Выберите признак:', vesh_cols)

    fig, ax = plt.subplots()
    ax.hist(df[feature], bins=30, color='#982d80')
    plt.title(f'Распределение значений по {feature}')
    st.pyplot(fig)
    
    st.subheader('Выводы')
    st.write('В основном в выборке люди от 20 до 40 лет')
    st.write('В основном в выборке люди с персональным доходом до 25 тысяч рублей')
    st.write('В основном первоначальный взнос по кредиту составляет до 20 тысяч рублей')
    st.write('В основном сумма кредита составляет до 20 тысяч рублей')
    
    st.subheader('Корреляционный анализ')
    st.image(correlation)
    st.write('Видим, что сильно скоррелированы между собой признаки величины кредита и первого взноса, срока кредита. С целевой переменной связь у признаков в основном не сильная')
        
    st.write('**Скоррелированные признаки**')
    st.image(credit_fstpayment)
    st.image(dependants_child)
        
    st.subheader('Связь признаков с таргетом')
    st.image(age_target)      
    st.image(education_target)
    st.image(income_target)
    st.image(gender_target)
