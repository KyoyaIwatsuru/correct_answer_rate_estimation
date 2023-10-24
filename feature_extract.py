#変更する点###
#
#use_file, path, confidence_answer_path, label_pathを変更する
#use_file, path, confidence_answer_path, label_pathは自分のパソコンでのファイル名などに変更
# 
#208行目，211行目のファイル名を変更
#
##############

# -*- coding: utf-8 -*-


import sys
import warnings
import os
import glob
import csv
import pandas as pd
import numpy as np


class MyClass():
	def __init__(self):
		self.x = []
		self.y = []
		self.z = []
		self.s_length = []
		self.s_angle = []
		self.s_velocity = []
		self.s_duration = []

		self.fixation_durations_choice = []
		self.fixation_durations_question = []
		self.fixation_durations_else = []
		self.attention = ""
		self.last_attention = ""
		self.transition = 0
		self.transition_in_question = 0
		self.transition_in_question_forward = 0
		self.transition_in_question_backward = 0
		self.transition_between_choice = 0
		self.transition_between_cq = 0
		self.transition_c_to_q = 0


def pre_process(fixations):
	X = []
	Y = []
	Z = []
	s_length = []
	s_angle = []
	s_velocity = []
	for i in range(len(fixations)):
		X.append(float(fixations[i][1])) #fixation_x
		Y.append(float(fixations[i][2])) #fixation_y
		Z.append(float(fixations[i][3])) #fixation_duration
		s_length.append(float(fixations[i][4]))
		s_angle.append(float(fixations[i][5]))
		s_velocity.append(float(fixations[i][6]))

	#print("X: ", len(X), len(X[0]))
	#print("Y: ", len(Y), len(Y[1]))
	#print("Z: ", len(Z), len(Z[2]))

	return X, Y, Z, s_length, s_angle, s_velocity


#Qa = [510, 288, 1410, 449]  #[left, top, right, bottom]
#Ca = [[530, 474, 950, 592], [960, 474, 1380, 592], [530, 602, 950, 720], [960, 602, 1380, 720]]


def feature_calculation(user, reading_time, correctness):
	f1 = len(user.fixation_durations_choice) #1:選択肢を見たfixationの総数
	f2 = len(user.fixation_durations_question) #2:問題文を見たfixationの総数
	if f1 == 0 and f2 == 0:
		f3 = 0
		f4 = 0
	else:
		f3 = 1. * f1 / (f1 + f2 + len(user.fixation_durations_else)) #3:全体のうち選択肢のfixationの割合
		f4 = 1. * f2 / (f1 + f2 + len(user.fixation_durations_else)) #4:全体のうち問題文のfixationの割合

	if len(user.fixation_durations_choice) == 0:
		user.fixation_durations_choice.append(0)
	if len(user.fixation_durations_question) == 0:
		user.fixation_durations_question.append(0)

	f5 = np.sum(user.fixation_durations_choice) #5:選択肢のfixation持続時間の合計
	f6 = np.mean(user.fixation_durations_choice) #6:選択肢のfixation持続時間の平均
	f7 = np.max(user.fixation_durations_choice) #7:選択肢のfixation持続時間の最大値
	f8 = np.min(user.fixation_durations_choice) #8:選択肢のfixation持続時間の最小値
	f9 = np.sum(user.fixation_durations_question) #9:問題文のfixation持続時間の合計
	f10 = np.mean(user.fixation_durations_question) #10:問題文のfixation持続時間の平均
	f11 = np.max(user.fixation_durations_question) #11:問題文のfixation持続時間の最大値
	f12 = np.min(user.fixation_durations_question) #12:問題文のfixation持続時間の最小値
	f13 = np.std(user.x) #13:x座標の分散値
	f14 = np.std(user.y) #14:y座標の分散値
	f17 = len(user.x) - 1 #17:saccadeの回数

	if len(user.x) - 1 == 0:
		user.s_length.append(0)
		user.s_duration.append(0)
		user.s_velocity.append(0)
		#print("len(user.x) - 1 = 0")

	f15 = np.sum(user.s_length) #15:saccadeの距離の合計
	f16 = np.mean(user.s_length) #16:saccadeの距離の平均

	f18 = user.transition_between_choice #18:選択肢間のsaccadeの回数
	f19 = user.transition_in_question #19:問題文内のsaccadeの回数
	f20 = user.transition_between_cq #20:選択肢-問題間のsaccadeの回数
	f21 = np.sum(user.s_duration) #21:saccade持続時間の合計
	f22 = np.mean(user.s_duration) #22:saccade持続時間の平均
	f23 = np.max(user.s_duration) #23:saccade持続時間の最大値
	f24 = np.min(user.s_duration) #24:saccade持続時間の最小値
	f25 = np.sum(user.s_velocity) #25:saccade時の視点の速度の合計
	f26 = np.mean(user.s_velocity) #26:saccade時の視点の速度の平均
	f27 = np.max(user.s_velocity) #27:saccade時の視点の速度の最大値
	f28 = np.min(user.s_velocity) #28:saccade時の視点の速度の最小値
	f29 = reading_time #29:解答時間
	f30 = correctness #30:解答の正誤

	return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11,
	f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30]


def fixation_detection(fixations, reading_time, correctness, user):
	X, Y, Z, s_length, s_angle, s_velocity = pre_process(fixations)

	#y_33 = (max(Y) - min(Y)) * 0.33 + min(Y)
	#y_66 = (max(Y) - min(Y)) * 0.66 + min(Y)
	#x_50 = (max(X) - min(X)) * 0.50 + min(Y)
	#問題文の範囲q:50 <= x <= 1850, 200 <= y <= 400
	x_ql = 50
	x_qr = 1850
	y_qu = 200
	y_qd = 400
	#選択肢の範囲c_all:50 <= x <= 500, c1:400 < y <= 519, c2:519 < y <= 611, c3:611 < y <= 703, c4:703 < y <= 800(1問おおよそ92区切り)
	x_cl = 50
	x_cr = 500
	y_c1 = 519
	y_c2 = 611
	y_c3 = 703
	y_c4 = 800

	user.x = []
	user.y = []
	user.z = []

	last_fixation = False
	lfx = -1
	lfy = -1
	is_last_in_question = False
	is_last_in_choice = False
	for i in range(len(X)): #fixation loop
		user.x.append(X[i]) #x
		user.y.append(Y[i])	#y
		user.z.append(Z[i])	#duration

		#where is current fixation on question or choice
		is_current_in_question = False
		is_current_in_choice = False
		if ((x_ql <= user.x[-1] <= x_qr) and (y_qu <= user.y[-1] <= y_qd)): #current fixation is on question
			is_current_in_question = True
			user.attention = "q"
			user.fixation_durations_question.append(user.z[-1])
		elif ((x_cl <= user.x[-1] <= x_cr) and (y_qd < user.y[-1] <= y_c1)): #current fixation is on choice
			is_current_in_choice = True
			user.attention = "c1"
			user.fixation_durations_choice.append(user.z[-1])
		elif ((x_cl <= user.x[-1] <= x_cr) and (y_c1 < user.y[-1] <= y_c2)): #current fixation is on choice
			is_current_in_choice = True
			user.attention = "c2"
			user.fixation_durations_choice.append(user.z[-1])
		elif ((x_cl <= user.x[-1] <= x_cr) and (y_c2 < user.y[-1] <= y_c3)): #current fixation is on choice
			is_current_in_choice = True
			user.attention = "c3"
			user.fixation_durations_choice.append(user.z[-1])
		elif ((x_cl <= user.x[-1] <= x_cr) and (y_c3 < user.y[-1] <= y_c4)): #current fixation is on choice
			is_current_in_choice = True
			user.attention = "c4"
			user.fixation_durations_choice.append(user.z[-1])
		else: #current fixation is on else
			user.attention = "e"
			user.fixation_durations_else.append(user.z[-1])

		#transition check
		if last_fixation:
			user.s_length.append(s_length[i])
			user.s_angle.append(s_angle[i])
			user.s_velocity.append(s_velocity[i])
			user.s_duration.append(s_length[i] / s_velocity[i])
			if (user.attention == "q") and (user.last_attention == "q"): #transition in question
				user.transition_in_question += 1
				if user.x[-1] > lfx:
					user.transition_in_question_forward += 1 #前のfixationから進んでいる
				elif user.x[-1] < lfx:
					user.transition_in_question_backward += 1 #前のfixationから戻っている
			elif (user.attention != "q") and (user.last_attention != "q") and (user.attention != "e") and (user.last_attention != "e") and (user.attention != user.last_attention): #transition between choice
				user.transition_between_choice += 1
			elif ((user.attention != "e") and (user.last_attention != "e") and (is_current_in_question and is_last_in_choice)) or ((user.attention != "e") and (user.last_attention != "e") and (is_current_in_choice and is_last_in_question)): #transition between question and choice
				user.transition_between_cq += 1
				if is_current_in_question:
					user.transition_c_to_q += 1

			user.transition += 1

		last_fixation = True
		lfx = user.x[-1]
		lfy = user.y[-1]
		is_last_in_question = is_current_in_question
		is_last_in_choice = is_current_in_choice
		user.last_attention = user.attention
		#print(user.attention)

	if len(user.x) > 1:
		features = feature_calculation(user, reading_time, correctness)
	else:
		features = []
		#print("len = 1")

	if float("inf") in features:
		features = []
		#print("inf")

	user.__init__()

	return features


def save(output_dir, id, user_id, document_id, question_id, choice, label, features):
	#特徴量リストの作成
	file_exist = False
	if os.path.isfile(os.path.join(output_dir, "feature.csv")):
		file_exist = True

	with open(os.path.join(output_dir, "feature.csv"), "a") as f:
		writer = csv.writer(f, lineterminator = '\n')
		head = ["#id", "user_id", "document_id", "question_id", "choice", "label", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11",
		"f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29", "f30"]

		if file_exist == False:
			writer.writerow(head)

		arr = [id, user_id, document_id, question_id, choice, label]
		arr.extend(features)

		writer.writerow(arr)

#def p(dir_name):
#	return os.path.abspath(os.path.expanduser(dir_name))

def main():

	use_file = ["\\guest\\", "\\pre01\\", "\\pre02\\", "\\p01\\", "\\p02\\", "\\p03\\", "\\p04\\", "\\p05\\", "\\p06\\", "\\p07\\", "\\p08\\", "\\p09\\", "\\p10\\"]

	for user in use_file:
		print(user)
		username = user
		if not sys.warnoptions:
			warnings.simplefilter("ignore")

		path = os.path.expanduser("W:\\Crowdsourcing")
		path += "\\loggerstation_iwatsuru"

		#要変更
		path += user

		input_dirs = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
		input_dirs.sort()

		for input_dir in input_dirs:
			#save_count = 0
			#print(input_dir)
			section_path = os.path.join(path, input_dir)

			if os.path.exists(os.path.join(section_path, "output")):
				if os.path.exists(os.path.join(section_path, "output\\feature.csv")):
					os.remove(os.path.join(section_path, "output\\feature.csv"))
				fixation_files = glob.glob(os.path.join(section_path, "output\\*_fixations.csv"))
				fixation_files.sort()

				confidence_answer_path = os.path.join(section_path, "confidence_answer.csv")
				label_path = os.path.join(section_path, "confidence_label.csv")

				#print(input_dir, len(fixation_files))
				if len(fixation_files) > 0:
					conf_df = pd.read_csv(confidence_answer_path)
					id_conf = conf_df["#id"].values.tolist()
					begin_time = conf_df["begin_timestamp"].values.tolist()
					end_time = conf_df["end_timestamp"].values.tolist()
					user_id = username
					document_id = input_dir #conf_df["document_id"].values.tolist()
					question_id = conf_df["question_id"].values.tolist()
					choice = conf_df["choice"]
					correctness = conf_df["correctness"].values.tolist()

					reading_time = []
					for i in range(len(begin_time)):
						reading_time.append(float(end_time[i]) - float(begin_time[i])) #sec

					label_df = pd.read_csv(label_path)
					id_label = label_df["#id"].values.tolist()
					label = label_df["confidence_label"].values.tolist()

					for fixation_path in fixation_files: #question loop
						fixation_df = pd.read_csv(fixation_path).dropna()
						#fixation = fixation_df.values.tolist()
						label_count = 0
						for i in range(len(id_conf)): #question loop
							if len(id_label) == 0:
								#print("len(id_label) = 0")
								continue
							if id_label[label_count] == id_conf[i]:
								df = fixation_df[(begin_time[i] < fixation_df["#timestamp"]) & (fixation_df["#timestamp"] < end_time[i])]
								fixations = df.values.tolist()

								if len(fixations) > 0:
									user = MyClass()
									features = fixation_detection(fixations, reading_time[i], correctness[i], user)

									if len(features) != 0:
										save(os.path.join(section_path, "output"), id_conf[i], user_id, document_id, question_id[i], choice[i], label[label_count], features)
										#save_count += 1
									#else:
										#print("len(features) = 0")

								label_count += 1
								if label_count >= len(id_label):
									break
			#print(save_count)


if __name__ == "__main__":
	main()