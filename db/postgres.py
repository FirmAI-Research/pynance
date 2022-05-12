import sys, os
import json

# conf
cwd = os.getcwd()
parentdir = os.path.dirname(cwd)
fp = os.path.join(parentdir, 'secrets.json')
with open(fp, 'r') as f:
    data = json.load(f)


from sqlalchemy import create_engine

class Postgres:

    def __init__(self):
        user = data['rds_user']
        password = data['rds_password']
        host = 'pynance.ckdvk4iveujs.us-east-1.rds.amazonaws.com'
        port = 5432
        database = 'postgres' 
        self.engine = create_engine(url="postgresql://{0}:{1}@{2}:{3}/{4}".format(user, password, host, port, database ) )

