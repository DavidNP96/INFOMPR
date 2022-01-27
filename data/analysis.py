import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
sns.color_palette("colorblind")

CONDITIONS = {
        'Q1':  'info',
        'Q2':  'real-0',
        'Q3':  'Baseline-1.5M',
        'Q4':  'Baseline-150K',
        'Q5':  'Baseline-75K',
        'Q6':  'Baseline-15K',
        'Q7':  'Retrained-1.5M',
        'Q8':  'Retrained-150K',
        'Q9':  'Retrained-75K',
        'Q10': 'Retrained-15K',
        'Q11': 'GPT2-1.5M',
        'Q12': "GPT2-150K",
        'Q13': "GPT2-75K",
        'Q14': "GPT2-15K"
}

MEASURES = {
    1: 'Appropriateness',
    2: 'Fluency',
    3: 'Coherence'
    
}

GENDER = {
    1: 'Male',
    2: 'Female',
    3: 'Other',
    4: 'Undisclosed'    
}

def clean_data(path = 'data.xlsx'):
    
    data = pd.read_excel(path)
    questions = [c for c in data.columns if c[0] == 'Q']
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
    
    columns = ['N', 'Gender', 'Age', 'Appropriateness', 'Fluency', 'Coherence', 'Model', 'Datasize']
    new_data = []

    for i in range(len(data)):
        datum = data.iloc[i]
        
        result = [datum[0], datum[1], datum[2]]
        
        for j in range(3, len(datum)):
            value = datum.iloc[j]
            condition = datum.index[j].split('_')[1]
            
            result.append(value)
            
            if (j-2)%3 == 0:
                result = result + condition.split('-')
                new_data.append(result)
                result = [datum[0], datum[1], datum[2]]
                
    return pd.DataFrame(new_data, columns = columns)



data = clean_data('data.xlsx')
data_by_condition = transpose_data(data)
data_no_real = data_by_condition[data_by_condition.Model != 'real']
data_real = data_by_condition[data_by_condition.Model == 'real']

real_appropriateness = data_real.Appropriateness.mean()
real_fluency = data_real.Fluency.mean()
real_coherence = data_real.Coherence.mean()

data_by_condition.groupby('Datasize').mean()
data_by_condition.groupby('Model').mean()
data_by_condition.groupby(['Model', 'Datasize']).mean()

data_by_condition.groupby('Datasize').std()
data_by_condition.groupby('Model').std()
data_by_condition.groupby(['Model', 'Datasize']).std()

model = ols('Appropriateness ~ C(Model) + C(Datasize) + C(Model):C(Datasize)', data=data_no_real).fit()
print(sm.stats.anova_lm(model, typ=2))

plt.axhline(real_coherence, ls='-.', color='black', label='Real')
plt.ylim(0, 8)
sns.lineplot(
    data=data_no_real, x="Datasize", y="Appropriateness", 
    hue="Model", style='Model', err_style="bars",
    markers=True, ms=10
)
plt.legend(bbox_to_anchor=(1, 1.1))
plt.savefig("Appropriateness.png", format="png", dpi=1200)
plt.show()
plt.clf()


model = ols('Fluency ~ C(Model) + C(Datasize) + C(Model):C(Datasize)', data=data_no_real).fit()
print(sm.stats.anova_lm(model, typ=2))

plt.axhline(real_fluency, ls='-.', color='black', label='Real')
plt.ylim(0, 8)
sns.lineplot(
    data=data_no_real, x="Datasize", y="Fluency", 
    hue="Model", style='Model', err_style="bars",
    markers=True, ms=10#, palette=sns.color_palette("colorblind")
)
plt.legend(bbox_to_anchor=(1, 1.1))
plt.savefig("Fluency.png", format="png", dpi=1200)
plt.show()
plt.clf()

model = ols('Coherence ~ C(Model) + C(Datasize) + C(Model):C(Datasize)', data=data_no_real).fit()
print(sm.stats.anova_lm(model, typ=2))

plt.axhline(real_coherence, ls='-.', color='black', label='Real')
plt.ylim(0, 8)
sns.lineplot(
    data=data_no_real, x="Datasize", y="Coherence", 
    hue="Model", style='Model', err_style="bars",
    markers=True, ms=10#, palette=['r', 'b', 'g']
)
plt.legend(bbox_to_anchor=(1, 1.1))
plt.savefig("Coherence.png", format="png", dpi=1200)
plt.show()
plt.clf()
