
import datetime as dt
import time

current_time = dt.datetime.now()
await_time = dt.datetime.now() + dt.timedelta(hours=2, minutes=40)
print("current time: " + current_time.strftime("%Y-%m-%d %H:%M:%S"))
print("stop time: " + await_time.strftime("%Y-%m-%d %H:%M:%S"))

while(current_time < await_time):
    time.sleep(5)
    current_time = dt.datetime.now()
    print("current time: " + current_time.strftime("%Y-%m-%d %H:%M:%S"))

print("done waiting")