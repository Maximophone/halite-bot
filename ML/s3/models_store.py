import boto
from datetime import datetime as dt

class ModelsStore(object):
    def __init__(self,bucket='mf-halite', models_folder='models'):
        self.bucket_name = bucket
        self.models_folder = models_folder

    def store_model(self,model,name):
        now = dt.now()
        s3 = boto.connect_s3()
        bucket = s3.get_bucket(self.bucket_name)
        key = bucket.new_key('{}/{}_{}.h5'.format(self.models_folder,name,now.strftime('%Y%m%d-%H:%M:%S')))
        latest_key = bucket.new_key('{}/{}_latest.h5'.format(self.models_folder,name))
        key.set_contents_from_filename(model)
        latest_key.set_contents_from_filename(model)
        s3.close()