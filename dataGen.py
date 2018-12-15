from datetime import datetime
import random
 
# generate random time
year = 2018
month = random.randint(1, 12)
day = random.randint(1, 28)
hour = random.randint(0,23)
minute = random.randint(0,59)
second = random.randint(0,59)
birth_date = datetime(year, month, day,1,3,2)
# print(birth_date.strftime("%Y-%m-%d %H:%M:%S"))

data = []

for i in range(500):
    temp = [random.randint(20,111), random.randint(20,100),birth_date.strftime("%Y-%m-%d %H:%M:%S"), "safe"]
    data.append(temp)

for i in range(500):
    temp = [random.randint(111,200), random.randint(20,100),birth_date.strftime("%Y-%m-%d %H:%M:%S"), "unsafe"]
    data.append(temp)

