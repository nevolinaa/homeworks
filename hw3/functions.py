from PIL import Image
import streamlit as st


def get_images():
    gender = Image.open('pictures/gender.png')
    flat = Image.open('pictures/flat.png')
    credit_status = Image.open('pictures/credit_status.png')
    working = Image.open('pictures/working.png')
    pens = Image.open('pictures/pens.png')
    auto = Image.open('pictures/auto.png')
    education = Image.open('pictures/education.png')
    family_status = Image.open('pictures/family_status.png')
    family_revenue = Image.open('pictures/family_revenue.png')

    target = Image.open('pictures/target.png')

    children = Image.open('pictures/children.png')
    term_credit = Image.open('pictures/term_credit.png')
    industry_work = Image.open('pictures/industry_work.png')
    work_title = Image.open('pictures/work_title.png')
    work_dir = Image.open('pictures/work_dir.png')

    age = Image.open('pictures/age.png')
    
    personal_income = Image.open('pictures/personal_income.png')
    personal_income_log = Image.open('pictures/personal_income_log.png')
    
    sum_credit = Image.open('pictures/sum_credit.png')
    sum_credit_log = Image.open('pictures/sum_credit_log.png')
    
    first_payment = Image.open('pictures/first_payment.png')
    first_payment_log = Image.open('pictures/first_payment_log.png')

    correlation = Image.open('pictures/correlation.png')

    credit_fstpayment = Image.open('pictures/credit_fstpayment.png')
    dependants_child = Image.open('pictures/dependants_child.png')

    age_target = Image.open('pictures/age_target.png')
    gender_target = Image.open('pictures/gender_target.png')
    education_target = Image.open('pictures/education_target.png')
    income_target = Image.open('pictures/income_target.png')
    
    return gender, flat, credit_status, working, pens, auto, education, family_status, family_revenue, target, \
           children, term_credit, industry_work, work_title, work_dir, age, personal_income, personal_income_log, sum_credit,\
           sum_credit_log, first_payment, first_payment_log, correlation, credit_fstpayment, dependants_child, age_target, \
           gender_target, education_target, income_target

def plot_features():
    gender, flat, credit_status, working, pens, auto, education, family_status, family_revenue, target, \
    children, term_credit, industry_work, work_title, work_dir, age, personal_income, personal_income_log, sum_credit,\
    sum_credit_log, first_payment, first_payment_log, correlation, credit_fstpayment, dependants_child, age_target, \
    gender_target, education_target, income_target = get_images()
        
    st.subheader('Графики распределения: таргет')
    st.image(target)
    st.write('В основном в выборке у клиентов не было отклика на маркетинговое предложение. Выборка несбалансирована')
        
    st.subheader('Графики распределения: категориальные переменные')
    st.image(gender)
    st.write('В выборке больше мужчин')
        
    st.image(flat)
    st.write('В выборке больше людей, которые не имеют собственную квартиру')
        
    st.image(credit_status)
    st.write('Большая часть кредитов в выборке закрыта')
        
    st.image(working)
    st.write('В основном люди в выборке работающие')
        
    st.image(pens)
    st.write('В основном люди в выборке не пенсионеры')
        
    st.image(auto)
    st.write('В основном люди в выборке не имеют машины')
        
    st.image(education)
    st.write('В основном люди в выборке с средним и средне-специальным образованием')
        
    st.image(family_status)
    st.write('В основном люди в выборке состоят в браке')
        
    st.image(family_revenue)
    st.write('В основном люди в выборке с доходами от 10 до 50 тысяч на семью')
        
    st.image(children)
    st.write('В выборке в основном представлены семьи с 1-2 детьми')
        
    st.image(term_credit)
    st.write('Видим, что есть несколько крупных групп: с кредитам на 3,6,10 и 12 лет')
        
    st.image(industry_work)
    st.write('В основном в выборке представлены работающие в торговле люди')
        
    st.image(work_title)
    st.write('В основном в выборке представлены специалисты и рабочие')
        
    st.image(work_dir)
        
    st.subheader('Графики распределения: вещественные переменные')
    st.image(age)
    st.write('В основном в выборке люди от 20 до 40 лет')
    
    st.write('**Графики до логарифмирования и после**')
    st.image(personal_income)
    st.image(personal_income_log)
    st.write('В основном в выборке люди с персональным доходом до 25 тысяч рублей')
    
    st.write('**Графики до логарифмирования и после**')
    st.image(first_payment)
    st.image(first_payment_log)
    st.write('В основном первоначальный взнос по кредиту составляет до 20 тысяч рублей')
    
    st.write('**Графики до логарифмирования и после**')
    st.image(sum_credit)
    st.image(sum_credit_log)
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