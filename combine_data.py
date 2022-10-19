import argparse
import json
import os

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dir1", type=str,
					default="/home/matej/Documents/metaphor-detection/data/komet")
parser.add_argument("--dir2", type=str,
					default="/home/matej/Documents/metaphor-detection/data/gkomet")
parser.add_argument("--combined_dir", type=str, default="combined-komet-gkomet")

if __name__ == "__main__":
	args = parser.parse_args()
	if os.path.exists(args.combined_dir):
		raise ValueError("combined_dir exists, so the data could accidentally get overriden."
						 "Please remove the dir manually and rerun the script")

	os.makedirs(args.combined_dir)

	with open(os.path.join(args.combined_dir, "config.json"), "w") as f:
		json.dump(vars(args), fp=f, indent=4)

	train_file1, dev_file1, test_file1 = None, None, None
	for fname in os.listdir(args.dir1):
		if not os.path.isfile(os.path.join(args.dir1, fname)):
			continue

		if "train" in fname:
			train_file1 = fname

		if "dev" in fname:
			dev_file1 = fname

		if "test" in fname:
			test_file1 = fname

	train_file2, dev_file2, test_file2 = None, None, None
	for fname in os.listdir(args.dir2):
		if not os.path.isfile(os.path.join(args.dir2, fname)):
			continue

		if "train" in fname:
			train_file2 = fname

		if "dev" in fname:
			dev_file2 = fname

		if "test" in fname:
			test_file2 = fname

	# Train
	if train_file1 is not None and train_file2 is not None:
		train_path1 = os.path.join(args.dir1, train_file1)
		train_path2 = os.path.join(args.dir2, train_file2)
		print(f"train_path1: {train_path1}\n"
			  f"train_path2: {train_path2}")

		df_train1 = pd.read_csv(train_path1, sep="\t")
		df_train2 = pd.read_csv(train_path2, sep="\t")

		combined_train = pd.concat((df_train1, df_train2), axis=0).reset_index(drop=True)

		print(f"Combined initial {df_train1.shape[0]} and {df_train2.shape[0]} into "
			  f"{combined_train.shape[0]} combined training instances\n")
		combined_train.to_csv(os.path.join(args.combined_dir, "train.tsv"), sep="\t", index=False)
	else:
		print("Warning: A training file could not be found in one or both of the directories, not combining training set\n")

	# Dev
	if dev_file1 is not None and dev_file2 is not None:
		dev_path1 = os.path.join(args.dir1, dev_file1)
		dev_path2 = os.path.join(args.dir2, dev_file2)
		print(f"dev_path1: {dev_path1}\n"
			  f"dev_path2: {dev_path2}")

		df_dev1 = pd.read_csv(dev_path1, sep="\t")
		df_dev2 = pd.read_csv(dev_path2, sep="\t")

		combined_dev = pd.concat((df_dev1, df_dev2), axis=0).reset_index(drop=True)

		print(f"Combined initial {df_dev1.shape[0]} and {df_dev2.shape[0]} into "
			  f"{combined_dev.shape[0]} combined dev instances\n")
		combined_dev.to_csv(os.path.join(args.combined_dir, "dev.tsv"), sep="\t", index=False)
	else:
		print("Warning: A dev file could not be found in one or both of the directories, not combining dev set\n")

	# Test
	if test_file1 is not None and test_file2 is not None:
		test_path1 = os.path.join(args.dir1, test_file1)
		test_path2 = os.path.join(args.dir2, test_file2)
		print(f"test_path1: {test_path1}\n"
			  f"test_path2: {test_path2}")

		df_test1 = pd.read_csv(os.path.join(args.dir1, test_file1), sep="\t")
		df_test2 = pd.read_csv(os.path.join(args.dir2, test_file2), sep="\t")

		combined_test = pd.concat((df_test1, df_test2), axis=0).reset_index(drop=True)

		print(f"Combined initial {df_test1.shape[0]} and {df_test2.shape[0]} into "
			  f"{combined_test.shape[0]} combined test instances\n")
		combined_test.to_csv(os.path.join(args.combined_dir, "test.tsv"), sep="\t", index=False)
	else:
		print("Warning: A test file could not be found in one or both of the directories, not combining test set\n")
