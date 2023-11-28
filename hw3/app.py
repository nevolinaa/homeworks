import pandas as pd
import streamlit as st
import time
from PIL import Image

from functions import plot_features
from model import pack_input, preprocess_data, open_data, predict_on_input

bank = Image.open('./pictures/bank.png')

_, _, _, _, scaler = preprocess_data(open_data('./data/df.csv'))

def render_page(bank):
    """ creates app page with tabs """

    st.title('Проверяем склонность клиента к отклику на маркетинговое предложение банка')
    st.subheader('Исследуем признаки, предсказываем склонность к отклику, оцениваем важность факторов')
    st.write('Материал - социально-экономические характеристики клиентов')
    st.image(bank)

    tab1, tab2, tab3 = st.tabs([':mag: Исследовать', ':crystal_ball: Предсказать', ':bar_chart: Оценить'])

    with tab1:
        st.subheader('EDA: исследуем данные')
        
        st.subheader('Описательные статистики: категориальные переменные')
        description_cat = pd.read_csv('./data/description_cat.csv', index_col='Unnamed: 0')
        st.dataframe(description_cat)

        st.subheader('Описательные статистики: вещественные переменные')
        description_num = pd.read_csv('./data/description_num.csv', index_col='Unnamed: 0')
        st.dataframe(description_num)
    
        plot_features()

    with tab2:
        st.write('Введите данные клиента:')

        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox('Пол', ['Женский', 'Мужской'])
            age = st.slider('Возраст', min_value=1, max_value=70)
        with col2:
            education = st.selectbox('Образование', ['Неполное среднее', 'Среднее', 'Среднее специальное', 'Неоконченное высшее', 'Высшее', 'Два и более высших образования', 'Ученая степень'])
            marital_status = st.selectbox('Семейное положение', ['Не состоял в браке', 'Гражданский брак', 'Состою в браке', 'Разведен(а)', 'Вдовец/Вдова'])
        with col3:
            child_total = st.slider('Количество детей', min_value=0, max_value=10)
            dependants = st.slider('Количество иждивенцев на попечении', min_value=0, max_value=10)
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            work_fl = st.selectbox('Есть ли работа?', ['Да', 'Нет'])
            pens_fl = st.selectbox('Находится ли на пенсии?', ['Да', 'Нет'])
            flat_fl = st.selectbox('Есть ли квартира?', ['Да', 'Нет'])

        with col2:
            auto = st.selectbox('Сколько автомобилей в собственности', [0, 1, 2])
            family_revenue = st.selectbox('Укажите доход на семью', ['до 5000 руб.', 'от 10000 до 20000 руб.', 'от 20000 до 50000 руб.', 'от 5000 до 10000 руб.', 'свыше 50000 руб.'])
            personal_revenue = st.slider('Укажите персональный доход', min_value=1, max_value=250000)
            
        with col3:
            job_type = st.selectbox('Укажите сферу работы', ['Другие сферы', 'Торговля', 'Государственная служба', 'Строительство', 'Металлургия/Промышленность/Машиностроение', 'Образование', 'Здравоохранение', 'Наука', 'Ресторанный бизнес/Общественное питание','СМИ/Реклама/PR-агенства', 'ЧОП/Детективная д-ть', 'Сельское хозяйство', 'Транспорт', 'Энергетика', 'Химия/Парфюмерия/Фармацевтика', 'Банк/Финансы', 'Информационные услуги', 'Коммунальное хоз-во/Дорожные службы', 'Нефтегазовая промышленность', 'Развлечения/Искусство', 'Сборочные производства', 'Салоны красоты и здоровья', 'Информационные технологии', 'Подбор персонала', 'Страхование', 'Юридические услуги/нотариальные услуги', 'Недвижимость', 'Туризм', 'Управляющая компания', 'Логистика', 'Маркетинг'])
            job_title = st.selectbox('Укажите должность на работе', ['Работник сферы услуг', 'Специалист', 'Руководитель высшего звена', 'Рабочий', 'Высококвалифиц. специалист', 'Служащий', 'Руководитель среднего звена', 'Индивидуальный предприниматель', 'Другое', 'Военнослужащий по контракту', 'Руководитель низшего звена', 'Партнер'])
            job_dir = st.selectbox('Укажите направление деятельности на работе', ['Участие в основ. деятельности', 'Бухгалтерия, финансы, планир.', 'Снабжение и сбыт', 'Вспомогательный техперсонал', 'Адм-хоз. и трансп. службы', 'Кадровая служба и секретариат', 'Служба безопасности', 'Реклама и маркетинг', 'Юридическая служба', 'Пр-техн. обесп. и телеком.'])
            work_time = st.slider('Укажите сколько клиент уже работает на текущем месте в месяцах', min_value=0, max_value=500)
        st.divider()
        
        st.write('Заполните информацию о кредите клиента:')    
        col1,col2 = st.columns(2)
        with col1:
            credit = st.slider('Сумма кредита', min_value=1, max_value=119700)
            term = st.slider('Срок кредита', min_value=3, max_value=36)
        with col2:
            first_payment = st.slider('Сколько составил первоначальный взнос по кредиту', min_value=0, max_value=140000)
            closed_fl = st.selectbox('Закрыт ли кредит?', ['Да', 'Нет'])
            
        if col2.button('Предсказать'):
            with st.spinner('Считаем'):
                time.sleep(1)
                inputs = pack_input(gender, age, education, marital_status, child_total, 
                                    dependants,  work_fl, pens_fl, flat_fl, auto, 
                                    family_revenue, personal_revenue, job_type, job_title, 
                                    job_dir, work_time, credit, term, first_payment, closed_fl)
                order_columns = ['AGE', 'GENDER', 'CHILD_TOTAL', 'DEPENDANTS', 'SOCSTATUS_WORK_FL',
                                 'SOCSTATUS_PENS_FL', 'FL_PRESENCE_FL', 'OWN_AUTO', 'TERM', 'WORK_TIME',
                                 'CLOSED_FL', 'LOG_FST_PAYMENT', 'LOG_PERSONAL_INCOME', 'LOG_CREDIT',
                                 'AGE_20_45', 'Два и более высших образования', 'Неоконченное высшее',
                                 'Неполное среднее', 'Среднее', 'Среднее специальное', 'Ученая степень',
                                 'Гражданский брак', 'Не состоял в браке', 'Разведен(а)',
                                 'Состою в браке', 'от 10000 до 20000 руб.', 'от 20000 до 50000 руб.',
                                 'от 5000 до 10000 руб.', 'свыше 50000 руб.']
                scaled = pd.DataFrame(scaler.transform(inputs[order_columns]), columns=inputs[order_columns].columns)
                pred = predict_on_input(scaled)
                
                if pred == 1:
                    st.success('Пассажир склонен к отклику на маркетинговое предложение')

                elif pred == 0:
                    st.error('Пассажир не откликнется на маркетинговое предложение')

                else:
                    st.error('Что-то пошло не так...')
                    
    with tab3:
        feature_importances = Image.open('./pictures/feature_importances.png')
        st.write('**Важные для модели признаки:**')
        st.image(feature_importances)
        
        shap = Image.open('./pictures/shap.png')
        st.write('**SHAP**')
        st.image(shap)
        
if __name__ == "__main__":
    render_page(bank)
