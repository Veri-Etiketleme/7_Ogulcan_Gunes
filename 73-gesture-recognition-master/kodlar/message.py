import webbrowser
import os
from collections import Counter
def message(key):
    if key==1:
        text='symbol1:need water'
    elif key==2:
        text='symbol2:hi'
    elif key==3:
        text='symbol3:bye bye'
    elif key==4:
        text='symbol4:hello'
    else:
        text='symbol5:see you'

    savefile=open('data.txt','w')
    savefile.write(text)
    savefile.close()
    webbrowser.open('http://localhost/fcmtest/send.php')
