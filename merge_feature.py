# -*- coding: utf-8 -*-


import sys
import warnings
import os
import csv


#def p(dir_name):
#	return os.path.abspath(os.path.expanduser(dir_name))

def main():
	use_file = ["\\guest\\", "\\pre01\\", "\\pre02\\", "\\p01\\", "\\p02\\", "\\p03\\", "\\p04\\", "\\p05\\", "\\p06\\", "\\p07\\", "\\p08\\", "\\p09\\", "\\p10\\"]

	for user in use_file:
		print(user)
		if not sys.warnoptions:
			warnings.simplefilter("ignore")

		path = os.path.expanduser("W:\\Crowdsourcing")
		path += "\\loggerstation_iwatsuru"

		#要変更
		path += user

		input_dirs = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
		input_dirs.sort()

		output_dir = path
		if os.path.exists(os.path.join(output_dir, "features.csv")):
			os.remove(os.path.join(output_dir, "features.csv"))

		#head = ["#id", "user_id", "document_id", "question_id", "choice", "label", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11",
		#"f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29", "f30"]
		features_all = []
		features_correct = []
		features_wrong = []

		for input_dir in input_dirs:
			#print(input_dir)
			section_path = os.path.join(path, input_dir)

			if os.path.exists(os.path.join(section_path, "output")):
				if os.path.exists(os.path.join(section_path, "output\\feature.csv")):
					feature_file = os.path.join(section_path, "output\\feature.csv")

					with open(feature_file, "r") as f:
						reader = csv.reader(f, lineterminator = "\n")
						header = next(reader)
						for row in reader:
							features_all.append(row)
							if row[-1] == "1":
								features_correct.append(row)
							else:
								features_wrong.append(row)

		print(len(features_all), len(features_correct), len(features_wrong))
		print(float(len(features_correct) / len(features_all)))

		if not os.path.exists(os.path.join(output_dir, "features.csv")):
			with open(os.path.join(output_dir, "features.csv"), "a") as f:
				writer = csv.writer(f, lineterminator = "\n")
				writer.writerow(header)

		with open(os.path.join(output_dir, "features.csv"), "a") as f:
			writer = csv.writer(f, lineterminator = "\n")
			writer.writerows(features_all)

		#if not os.path.exists(os.path.join(output_dir, "feature_correct.csv")):
		#	with open(os.path.join(output_dir, "feature_correct.csv"), "a") as f:
		#		writer = csv.writer(f, lineterminator = "\n")
		#		writer.writerow(head)

		#with open(os.path.join(output_dir, "feature_correct.csv"), "a") as f:
		#	writer = csv.writer(f, lineterminator = "\n")
		#	writer.writerows(features_correct)

		#if not os.path.exists(os.path.join(output_dir, "feature_wrong.csv")):
		#	with open(os.path.join(output_dir, "feature_wrong.csv"), "a") as f:
		#		writer = csv.writer(f, lineterminator = "\n")
		#		writer.writerow(head)

		#with open(os.path.join(output_dir, "feature_wrong.csv"), "a") as f:
		#	writer = csv.writer(f, lineterminator = "\n")
		#	writer.writerows(features_wrong)


if __name__ == "__main__":
	main()