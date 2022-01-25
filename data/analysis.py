import pandas as pd
import numpy as np

CONDITIONS = {
        'Q1':  'info',
        'Q2':  'real',
        'Q3':  'rnn100',
        'Q4':  'rnn50',
        'Q5':  'rnn10',
        'Q6':  'rnn1',
        'Q7':  're100',
        'Q8':  're50',
        'Q9':  're10',
        'Q10': 're1',
        'Q11': 'gpt100',
        'Q12': "gpt50",
        'Q13': "gpt10",
        'Q14': "gpt1"
}

MEASURES = {
    1: 'appropriateness',
    2: 'fluency',
    3: 'coherence'
    
}

GENDER = {
    1: 'male',
    2: 'female',
    3: 'other',
    4: 'not disclosed'    
}

def clean_data(data, questions):
    
    columns = ['n', 'gender', 'age']
    new_data = []

    for i in range(len(data)):
        
        datum = data.iloc[i]
        result = [i]
        
        if datum.Finished == 1:
            
            result.append(GENDER[int(datum['Q1.2'])]) #gender
            result.append(int(datum['Q1.3'])) #age
            
            d = datum[questions]
            
            for j in range(3, len(d)):
                
                value = d.iloc[j]
                if not np.isnan(value):
                    
                    index = d.index[j]
                    
                    condition = CONDITIONS[index[:int(index.find('.'))]]
                    measure = MEASURES[int(index[int(index.find('_'))+1:])]
                    
                    result.append(value)
                    if i == 1:
                        columns.append(measure+'_'+condition)
                    
            new_data.append(result)
            
    return pd.DataFrame(new_data, columns=columns)

def transpose_data(data):
    
    columns = ['n', 'gender', 'age', 'appropriateness', 'fluency', 'coherence', 'condition']
    new_data = []

    for i in range(len(data)):
        datum = data.iloc[i]
        
        result = [datum[0], datum[1], datum[2]]
        
        for j in range(3, len(datum)):
            value = datum.iloc[j]
            condition = datum.index[j].split('_')[1]
            
            result.append(value)
            
            if (j-2)%3 == 0:
                result.append(condition)
                new_data.append(result)
                result = [datum[0], datum[1], datum[2]]
                
    return pd.DataFrame(new_data, columns = columns)

data = pd.read_excel('data.xlsx')
questions = [c for c in data.columns if c[0] == 'Q']

data = clean_data(data, questions)
data_by_condition = transpose_data(data)






