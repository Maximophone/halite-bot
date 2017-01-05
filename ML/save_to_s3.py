from s3.models_store import ModelsStore
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('model',type=str)
	parser.add_argument('name',type=str, nargs='?')

	args = parser.parse_args()

	ms = ModelsStore()
	ms.store_model(
		args.model,
		args.name if args.name is not None else args.model.split('/')[-1].rstrip('.h5')
		)
