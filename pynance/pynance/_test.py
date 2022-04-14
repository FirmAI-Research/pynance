import sys,os
lib_dirp = os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__))) )) # NOTE: relative path from manage.py to lib/
sys.path.append(lib_dirp) 

# import lib.learn.rnn.rnn as rnn
# rnn.init()

# import lib.learn.rnn.rnn as rnn
