import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import statistics as s
import random
import matplotlib.dates as mdates

from datetime import timedelta
from datetime import datetime
def main():
    data = pd.read_csv("data/total_dataset.csv")
    data['datetime']=  pd.to_datetime(data['datetime'])
    state_date = "2018-04-02"
    state_date = datetime.strptime(state_date, "%Y-%m-%d")
    day = (data['datetime'] >= state_date) & (data['datetime'] < (state_date + timedelta(days=1)))
    state = data.loc[day]
    state = data.loc[day]
    state_time=state['datetime']
    state_time.reset_index(drop=True, inplace=True)
    state_value=state['fwts']
    state_value.reset_index(drop=True, inplace=True)
    max_value = state_value.max()
    max_index = state_value.idxmax()
    #Discharge action
    discharge_1=max_index+4
    discharge_2=max_index-3
    discharge_3=max_index-8
    #4,3,8
    #experiment 1 , 5, 13,24
    #Shaved consumption calculation
    profit=0
    rate= 1/10**8
    capacity = 2000 #2000kwH
    discharge_hour=1

    Shaved_value = state_value.copy()
    Shaved_value[discharge_1:discharge_1+12] = Shaved_value[discharge_1:discharge_1+12]-capacity/discharge_hour
    Shaved_value[discharge_2:discharge_2+12] = Shaved_value[discharge_2:discharge_2+12]-capacity/discharge_hour
    Shaved_value[discharge_3:discharge_3+12] = Shaved_value[discharge_3:discharge_3+12]-capacity/discharge_hour

    #Profit calculation

    profit += rate*capacity*(s.mean(Shaved_value[discharge_1:discharge_1+12]))
    profit += rate*capacity*(s.mean(Shaved_value[discharge_2:discharge_2+12]))
    profit += rate*capacity*(s.mean(Shaved_value[discharge_3:discharge_3+12]))

    peak_shave = max_value - Shaved_value.max()
    #Plotting
    random.seed(1234)

    peak_shave = state_value.max() - Shaved_value.max()
    print(peak_shave)
    fig, ax = plt.subplots(1)
    fig.autofmt_xdate()
    xfmt = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(xfmt)
    plt.xlabel("Time (Hour:Minitue)")
    plt.ylabel("Consumption(W)")
    #plt.title("Peak Shaved: {1}W".format(reward, int(2764)))
    plt.title("Total Peak Shaved: 1804W")
    #plt.title("Discharging at same point: 499W Peak Shaved")
    Shaved, = plt.plot(state_time, Shaved_value,'y',label="Shaved Consumption")
    Acutal, = plt.plot(state_time, state_value,'r',label="Acutal Consumption")
    discharge_start_1, =plt.plot(state_time[discharge_1], state_value[discharge_1], 'b*')
    discharge_start_2, =plt.plot(state_time[discharge_2], state_value[discharge_2], 'b*')
    discharge_start_3, =plt.plot(state_time[discharge_3], state_value[discharge_3], 'b*')
    plt.legend([Shaved,Acutal,discharge_start_3], ["Shaved Consumption","Acutal Consumption","discharge start"])
    plt.show()

    agent_size = pd.read_csv("data/agent_size.csv")

    plt.xlabel("agent sizes")
    plt.ylabel("time / hours")
    plt.title("Relationship between training time and agent size")
    plt.plot(agent_size.agents, agent_size.hours,'r')
    plt.show()

    epoch = np.arange(0,2001,100)
    import multiprocessing as mp
    agents=agent_size.agents
    for n in agents:
        location = 'data/agent_number'+str(n)+'.csv'
        agent_number_time = pd.read_csv(location)
        #plt.plot(agent_number_time.epoch, agent_number_time.hours, label="agent number = %d"%(n,))
        #plt.legend()
    #plt.xlabel("epochs")
    #plt.ylabel("time / hours")
    #plt.title("Relationship between training time and epochs")
    #plt.show()

    parallel_programming = pd.read_csv('data/parallel_programming.csv')
    serial_programming = pd.read_csv('data/serial_programming.csv')

    #plt.plot(serial_programming.agents, serial_programming.hours, label="serial programming")
    #plt.plot(parallel_programming.agents, parallel_programming.hours, label="parallel programming")

    #plt.legend()
    #plt.xlabel("agent sizes")
    #plt.ylabel("time / hours")
    #plt.title("Comparison of parallel programming and serial programming")
    #plt.show()



    action_space_resolution = pd.read_csv('data/action_space_resolution.csv')
    plt.plot(action_space_resolution.action_space, action_space_resolution.action,"r")
    plt.xlabel("Action space resolution(min)")
    plt.ylabel("time / hours")
    plt.title("The effect of action space resolution to training time")
    plt.show()


    peak_shave_eplison_990 = pd.read_csv('data/peak_shave_epsilon0.990.csv')
    peak_shave_eplison_996 = pd.read_csv('data/peak_shave_epsilon0.996.csv')
    peak_shave_eplison_997 = pd.read_csv('data/peak_shave_epsilon0.997.csv')
    peak_shave_eplison_998 = pd.read_csv('data/peak_shave_epsilon0.998.csv')

    #print(peak_shave_list_7)
    plt.plot(peak_shave_eplison_990,label="$Exploration Decay(\epsilon)$ = 0.990",color ="darkred")
    plt.plot(peak_shave_eplison_996,label="$Exploration Decay(\epsilon)$ = 0.996",color ="red")
    plt.plot(peak_shave_eplison_997,label="$Exploration Decay(\epsilon)$ = 0.997",color ="salmon")
    plt.plot(peak_shave_eplison_998,label="$Exploration Decay(\epsilon)$ = 0.998",color = "gold")
    plt.xlabel("Epochs")
    plt.ylabel("Average Peak Shaved(W)")
    plt.title("Average Peak Shaved(W) for each epoch on different $\epsilon$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
        main()