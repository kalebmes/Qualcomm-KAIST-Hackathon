import csv
import pandas as pd

with open('first.csv', 'r') as f:
    reader = csv.reader(f)
    data1 = list(reader)
    
with open('second.csv', 'r') as f:
    reader = csv.reader(f)
    data2 = list(reader)
    
with open('third.csv', 'r') as f:
    reader = csv.reader(f)
    data3 = list(reader)
    

new_data = [['User_ID', 'I/E', 'S/N', 'T/F', 'J/P']]

# all the ensembles can be applied by the below mentioned code, the hardest one is given, the rest is just some minor changes

for j in range(1, len(data1)):
    new_temp = [str(data1[j][0])]
    for i in range(1, 5):
        temp1, temp2, temp3 = 0, 0, 0
        if float(data1[j][i]) > 0.55:
            temp1 = 1
        elif float(data1[j][i]) < 0.45:
            temp1 = -1
        else:
            temp1 = 0
            
        if float(data2[j][i]) > 0.55:
            temp2 = 1
        elif float(data2[j][i]) < 0.45:
            temp2 = -1
        else:
            temp2 = 0

        if float(data3[j][i]) > 0.55:
            temp3 = 1
        elif float(data3[j][i]) < 0.45:
            temp3 = -1
        else:
            temp3 = 0
            
        if  temp1 + temp2 + temp3 >= 2:
            new_temp.append(max(float(data1[j][i]), float(data2[j][i]), float(data3[j][i])))
        elif temp1 + temp2 + temp3 <= -2:
            new_temp.append(min(float(data1[j][i]), float(data2[j][i]), float(data3[j][i])))
        else:
            neww = [float(data1[j][i]), float(data2[j][i]), float(data3[j][i])]
            neww_abs = [abs(x-0.5) for x in neww]
            idxxx = neww_abs.index(min(neww_abs))
            new_temp.append(neww[idxxx])
        
    new_data.append(new_temp)
    
    
with open('ffens_new_proportion_new.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(new_data)
    
    
        
    
    
    
