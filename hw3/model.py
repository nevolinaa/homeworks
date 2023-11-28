import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pickle import load

def open_data(path="data/df.csv"):
    """ reads df from given path """
    df = pd.read_csv(path)

    return df

def pack_input(gender, age, education, marital_status, child_total, 
               dependants,  status_work, status_pens, status_flat, auto, 
               family_revenue, personal_revenue, job_type, work, 
               job_dir, work_time, credit, term, first_payment, closed_fl):
    """ translate input values to pass to model """

    rule = {'Женский': 0,
            'Мужской': 1,
            'Да' : 1,
            'Нет': 0}
    
    if age <= 45:
        age_20_45 = 1 
    else:
        0
        
    data = {'AGE': age, 
            'GENDER': rule[gender], 
            'CHILD_TOTAL': child_total, 
            'DEPENDANTS': dependants, 
            'SOCSTATUS_WORK_FL': rule[status_work],
            'SOCSTATUS_PENS_FL': rule[status_pens], 
            'FL_PRESENCE_FL': rule[status_flat], 
            'OWN_AUTO': auto, 
            'TERM': term, 
            'WORK_TIME': work_time,
            'CLOSED_FL': rule[closed_fl], 
            'LOG_FST_PAYMENT': np.log(first_payment), 
            'LOG_PERSONAL_INCOME':  np.log(personal_revenue), 
            'LOG_CREDIT': np.log(credit),
            'AGE_20_45': age_20_45, 
            'Высшее': 0,
            'Два и более высших образования': 0, 
            'Неоконченное высшее': 0,
            'Неполное среднее': 0, 
            'Среднее': 0, 
            'Среднее специальное': 0, 
            'Ученая степень': 0,
            'Гражданский брак': 0, 
            'Не состоял в браке': 0, 
            'Разведен(а)': 0,
            'Состою в браке': 0, 
            'Вдовец/Вдова': 0,
            'до 5000 руб.': 0,
            'от 10000 до 20000 руб.': 0, 
            'от 20000 до 50000 руб.': 0,
            'от 5000 до 10000 руб.': 0, 
            'свыше 50000 руб.': 0}
    
    data[education] = 1
    data[marital_status] = 1
    data[family_revenue] = 1
    
    df = pd.DataFrame(data, index=[0])
    df.drop(columns=['Высшее', 'до 5000 руб.', 'Вдовец/Вдова'], axis=1, inplace=True)

    return df

def preprocess_data(df: pd.DataFrame):
    """ runs preprocessing on dataset """
    
    mode_fillna = df[['FAMILY_INCOME', 'TERM', 'GEN_INDUSTRY',
                  'GEN_TITLE', 'JOB_DIR', 'CLOSED_FL']].mode()
    median_fillna = df[['PERSONAL_INCOME', 'CREDIT', 'FST_PAYMENT', 'WORK_TIME']].median()

    df[['FAMILY_INCOME', 'TERM', 'GEN_INDUSTRY',
        'GEN_TITLE', 'JOB_DIR', 'CLOSED_FL']] = df[['FAMILY_INCOME', 'TERM', 'GEN_INDUSTRY',
                                                    'GEN_TITLE', 'JOB_DIR', 'CLOSED_FL']].fillna(mode_fillna)

    df[['PERSONAL_INCOME', 'CREDIT',
        'FST_PAYMENT', 'WORK_TIME']] = df[['PERSONAL_INCOME', 'CREDIT',
                                       'FST_PAYMENT', 'WORK_TIME']].fillna(median_fillna)
    
    df['LOG_CREDIT'] = np.log(df['CREDIT'])
    df['LOG_FST_PAYMENT'] = np.log(df['FST_PAYMENT'])
    df['LOG_PERSONAL_INCOME'] = np.log(df['PERSONAL_INCOME'])
    
    df['AGE_20_45'] = [1 if i <= 45 else 0 for i in df['AGE']]
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    X = df[['AGE', 'GENDER', 'EDUCATION', 'MARITAL_STATUS', 'CHILD_TOTAL',
        'DEPENDANTS', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL',
        'FL_PRESENCE_FL', 'OWN_AUTO', 'FAMILY_INCOME', 'TERM',
        'WORK_TIME', 'CLOSED_FL', 'LOG_FST_PAYMENT',
        'LOG_PERSONAL_INCOME', 'LOG_CREDIT', 'AGE_20_45']]

    y = df['TARGET']
    
    enc_class = pd.get_dummies(X['MARITAL_STATUS'], drop_first=True)
    X = pd.concat([X, enc_class], axis=1)
    X.drop(columns=['MARITAL_STATUS'], axis=1, inplace=True)
    
    enc_class = pd.get_dummies(X['EDUCATION'], drop_first=True)
    X = pd.concat([X, enc_class], axis=1)
    X.drop(columns=['EDUCATION'], axis=1, inplace=True)
    
    enc_class = pd.get_dummies(X['FAMILY_INCOME'], drop_first=True)
    X = pd.concat([X, enc_class], axis=1)
    X.drop(columns=['FAMILY_INCOME'], axis=1, inplace=True)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # масштабирование
    ss = MinMaxScaler()
    order_columns = ['AGE', 'GENDER', 'CHILD_TOTAL', 'DEPENDANTS', 'SOCSTATUS_WORK_FL',
                     'SOCSTATUS_PENS_FL', 'FL_PRESENCE_FL', 'OWN_AUTO', 'TERM', 'WORK_TIME',
                     'CLOSED_FL', 'LOG_FST_PAYMENT', 'LOG_PERSONAL_INCOME', 'LOG_CREDIT',
                     'AGE_20_45', 'Два и более высших образования', 'Неоконченное высшее',
                     'Неполное среднее', 'Среднее', 'Среднее специальное', 'Ученая степень',
                     'Гражданский брак', 'Не состоял в браке', 'Разведен(а)',
                     'Состою в браке', 'от 10000 до 20000 руб.', 'от 20000 до 50000 руб.',
                     'от 5000 до 10000 руб.', 'свыше 50000 руб.']
    ss.fit(X_train[order_columns])

    X_train = pd.DataFrame(ss.transform(X_train[order_columns]), columns=X_train[order_columns].columns)
    X_test = pd.DataFrame(ss.transform(X_test[order_columns]), columns=X_test[order_columns].columns)

    return X_train, X_test, y_train, y_test, ss

def loadmodel(path="hw3/model.mw"):
    """ load model"""

    with open(path, "rb") as file:
        model = load(file)

    return model

def predict_on_input(df):
    """ loads model and returns prediction """

    model = loadmodel()
    cols_when_model_builds = ['AGE', 'GENDER', 'CHILD_TOTAL', 'DEPENDANTS', 'SOCSTATUS_WORK_FL', 
                              'SOCSTATUS_PENS_FL', 'FL_PRESENCE_FL', 'OWN_AUTO', 'TERM', 'WORK_TIME', 
                              'CLOSED_FL', 'LOG_FST_PAYMENT', 'LOG_PERSONAL_INCOME', 'LOG_CREDIT', 
                              'AGE_20_45', 'Два и более высших образования', 'Неоконченное высшее', 
                              'Неполное среднее', 'Среднее', 'Среднее специальное', 'Ученая степень', 
                              'Гражданский брак', 'Не состоял в браке', 'Разведен(а)', 'Состою в браке', 
                              'от 10000 до 20000 руб.', 'от 20000 до 50000 руб.', 'от 5000 до 10000 руб.', 
                              'свыше 50000 руб.']
    data = df[cols_when_model_builds]
    pred = model.predict(data)

    return pred
