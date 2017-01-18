from datetime import datetime, time
from time import sleep

# def act(x):
#     return x+10

# def wait_start(runTime, action):
#     startTime = time(*(map(int, runTime.split(':'))))
#     while startTime > datetime.today().time(): # you can add here any additional variable to break loop if necessary
#         sleep(60)
# return action

def wait_till(runTime):
    startTime = time(*(map(int, runTime.split(':'))))
    while startTime > datetime.today().time(): # you can add here any additional variable to break loop if necessary
        sleep(20)

# wait_start('15:20', lambda: act(100))