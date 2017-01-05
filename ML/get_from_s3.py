from s3.models_store import ModelsStore
import argparse
import os

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('name',type=str)

	args = parser.parse_args()

	if os.path.isfile(args.name+'.h5'):
		if raw_input('{} already exists. Continue? (y/n)'.format(args.name+'.h5')).lower() not in ('y','yes'):
			exit(0)

	ms = ModelsStore()
	ms.get_latest_model(args.name)
