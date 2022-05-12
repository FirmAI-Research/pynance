import sys, os
import json

# conf
cwd = os.getcwd()
parentdir = os.path.dirname(cwd)
fp = os.path.join(parentdir, 'secrets.json')
with open(fp, 'r') as f:
    data = json.load(f)


from sqlalchemy import create_engine

user = data['rds_user']
password = data['rds_password']
host = 'pynance.ckdvk4iveujs.us-east-1.rds.amazonaws.com'
port = 5432
database = 'pynance'
  
def get_connection():
    return create_engine(
        url="mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(
            user, password, host, port, database
        )
    )
  
  
if __name__ == '__main__':
    try:        
        engine = get_connection()
        print(
            f"Connection to the {host} for user {user} created successfully.")
    except Exception as ex:
        print("Connection could not be made due to the following error: \n", ex)