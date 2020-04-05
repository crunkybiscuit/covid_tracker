#Python 3.6
from pandas import DataFrame,Series
import pandas as pd#0.23.0
import matplotlib.pyplot as plt#2.2.2
import seaborn as sns#0.8.1
from datetime import date

#pulls US Census data and Covid Tracking Project data
pop_df = pd.read_csv('http://www2.census.gov/programs-surveys/popest/datasets/2010-2019/national/totals/nst-est2019-alldata.csv?#')
test_df = pd.read_csv('http://covidtracking.com/api/states/daily.csv')

#converts US Census state numbers into state postal codes
def generate_pop_dict(raw_csv):
    state_dict = {1:'AL',2:'AK',4:'AZ',5:'AR',6:'CA',8:'CO',9:'CT',10:'DE',
                  11:'DC',12:'FL',13:'GA',15:'HI',16:'ID',17:'IL',18:'IN',
                  19:'IA',20:'KS',21:'KY',22:'LA',23:'ME',24:'MD',25:'MA',
                  26:'MI',27:'MN',28:'MS',29:'MO',30:'MT',31:'NE',32:'NV',
                  33:'NH',34:'NJ',35:'NM',36:'NY',37:'NC',38:'ND',39:'OH',
                  40:'OK',41:'OR',42:'PA',44:'RI',45:'SC',46:'SD',47:'TN',
                  48:'TX',49:'UT',50:'VT',51:'VA',53:'WA',54:'WV',55:'WI',
                  56:'WY',72:'PR'}
    just_states = raw_csv[raw_csv['STATE'] != 0]
    just_states.index = [state_dict[state] for state in just_states['STATE']]
    return just_states['POPESTIMATE2019'].to_dict()

#helper function for prep_covid_data()
def date_convert(date_int):
    year = int(str(date_int)[0:4])
    month = int(str(date_int)[4:6])
    day = int(str(date_int)[6:8])
    return date(year,month,day)

def prep_covid_data(raw_csv):
    int_date_range = raw_csv['date'].unique().tolist()
    datetime_dict = {date:date_convert(date) for date in int_date_range}
    return raw_csv.replace({'date':datetime_dict})


state_pop = generate_pop_dict(pop_df)        
covid_data = prep_covid_data(test_df)    

class State_Data_Generator(object):
    def __init__(self,state_population,covid_project_data,pop_adj=False):
        self.state_pop = state_population
        self.covid_data = covid_project_data
        self.pop_adj = pop_adj
        self.state_outbreak_date = {}
        self.days_since_100 = {}
        self.state_test_rate = {}
        self.state_growth_rate = {}
        self.state_positive_rate = {}
        self.state_death_rate = {}
        self.growth_curves = DataFrame(index=range(1,101),columns=[list(self.state_pop.keys())])

    #identifies the date at which each state reached 100 confirmed cases
    def outbreak_date(self,df,state):
        outbreak = df[df['positive'] >= 100]
        outbreak_date = outbreak['date'].min()
        try:
            days_since = (date.today() - outbreak_date).days
        except TypeError:
            days_since = 0
        self.state_outbreak_date[state] = outbreak_date
        self.days_since_100[state] = days_since      

    #calculates the cumulative testing rate as tests per million people
    def test_rate(self,df,state):
        test_rate = df['total'] * 1000000 / self.state_pop[state]
        self.state_test_rate[state] = Series(test_rate.values,index=df['date'])        

    #calculates growth rate over latest 3 day period for smoothing purposes
    def growth_rate(self,df,state):
        roll = df['positive'].rolling(4,min_periods=4)
        three_day = ((roll.max() - roll.min()) / roll.min() + 1) ** (1/3) - 1
        self.state_growth_rate[state] = Series(three_day.values,index=df['date'])        

    #calculates positive test rate over latest 3 day period for smoothing purposes
    def positive_rate(self,df,state):
        pos_roll = df['positive'].rolling(4,min_periods=4)
        test_roll = df['total'].rolling(4,min_periods=4)
        three_raw = (pos_roll.max() - pos_roll.min()) / (test_roll.max() - test_roll.min())
        self.state_positive_rate[state] = Series(three_raw.values,index=df['date'])    

    #calculates death rate over latest 3 day period for smoothing purposes        
    def death_rate(self,df,state):
        roll = df['death'].rolling(4,min_periods=4)
        three_day = ((roll.max() - roll.min()) / roll.min() + 1) ** (1/3) - 1
        self.state_death_rate[state] = Series(three_day.values,index=df['date']) 

    #calculates the aggregate case count for all states normalized to Day 1 of outbreak
    def growth_curve_calc(self,df,state,pop_adj=False):
        if self.state_outbreak_date[state] == 0:
            pass
        else:
            start_date = self.state_outbreak_date[state]
            state_100 = df[df['date'] >= start_date].sort_values('date')['positive']
            state_100.name = state
            state_100.index = range(1,len(state_100)+1)
            if pop_adj:
                state_100 /= (self.state_pop[state] / 1000000.0)
            self.growth_curves[state] = state_100

    #batch function running all of the above calculations for all states        
    def data_calc(self):
        for state in self.state_pop:
            curr_state = self.covid_data[self.covid_data['state'] == state]            
            subset = curr_state.sort_values('date')[['date','positive','death','total']]
            self.outbreak_date(subset,state)
            self.test_rate(subset,state)
            self.growth_rate(subset,state)
            self.positive_rate(subset,state)
            self.death_rate(subset,state)
            self.growth_curve_calc(subset,state,pop_adj=self.pop_adj)
        self.growth_curves = self.growth_curves.dropna(axis='rows',how='all').fillna('')      

    #generates dot plot using whichever two metrics calculated above    
    def generate_dot_plot(self, x_var, y_var, x_label, y_label, state_list=None):
        if state_list == None:
            state_list = list(self.state_pop.keys())
        else:
            state_list = state_list
        pos_test_plot = DataFrame(index=state_list,columns=[x_label, y_label,
                                                            'Days since 100th Case'])
        pos_test_plot['Days since 100th Case'] = [self.days_since_100[state] for state in state_list]
        pos_test_plot[x_label] = [x_var[state].iloc[-1] for state in state_list]
        pos_test_plot[y_label] = [y_var[state].iloc[-1] for state in state_list]
        print(pos_test_plot)

        sns.set()
        #size and color based on time since 100
        states = pos_test_plot.index.tolist()
        x_coords = pos_test_plot[x_label]
        y_coords = pos_test_plot[y_label]
        day_count = pos_test_plot['Days since 100th Case']
 
        for i,state in enumerate(states):
            x = x_coords[i]
            y = y_coords[i]
            days_since = day_count[i]
            if days_since >= 21:
                code = 'red'
            elif 21 > days_since >= 14:
                code = 'orange'
            elif 14 > days_since >= 7:
                code = 'yellow'
            elif 7 > days_since >= 1:
                code = 'greenyellow'
            else:
                code = 'green'
            plt.scatter(x, y, marker='o', color=code, s=days_since**2) 
            plt.text(x, y, state, fontsize=13,ha='left',va='bottom')
        plt.title("%s vs %s as of %s" %(x_label,y_label,str(date.today())))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
        
us = State_Data_Generator(state_pop,covid_data,pop_adj=True)
us.data_calc()
us.generate_dot_plot(us.state_test_rate,us.state_positive_rate,'testing','positive rate')
