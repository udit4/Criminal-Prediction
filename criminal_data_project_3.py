
from __future__ import division
import pandas as pd
from collections import Counter
from math import log10

# Prediction and accuracy using Naive Bayes Model.

path = '/Users/home/Desktop{}'

def data_preparation(input_file_path):
	input_df = pd.read_csv(input_file_path)
	input_df.drop(['ANALWT_C'],axis=1, inplace=True)
	# Feature Engineering Section.
	columns_to_remove = ['irki17_2','iihh65_2', 'iimedicr', 'iimcdchp', 'iichmpus', 'prvhltin', 'grphltin',
					'hltinnos', 'iiothhlt', 'irinsur4', 'hllosrsn', 'hlnvoffr', 'hlnvcost', 'hlnvneed', 'hlnvsor',
					'anyhlti2', 'govtprog', 'irwelmos', 'irfamin3', 'troubund', 'coutyp2', 'aiind102', 'prxydata',
					'cellwrkng', 'othins']
	columns_to_remove = [i.upper() for i in columns_to_remove]
	input_df.drop(columns_to_remove, axis=1, inplace=True)
	columns = list(input_df.columns)
	columns = [i.strip() for i in columns]
	input_df.columns = columns
	return input_df

def criminal_dictionary(input_file_path):
	criminal_dic = {}
	input_df = data_preparation(input_file_path)
	input_df = input_df[input_df['Criminal']==1]
	for i in input_df.columns:
		if(i in ['PERID','Criminal']):
			continue
		else:
			criminal_dic[i] = dict(Counter(input_df.loc[:,i]))
	return criminal_dic

def non_criminal_dictionary(input_file_path):
	non_criminal_dic = {}
	input_df = data_preparation(input_file_path)
	input_df = input_df[input_df['Criminal']==0]
	for i in input_df.columns:
		non_criminal_dic[i] = dict(Counter(input_df.loc[:,i]))
	return non_criminal_dic


def probability_criminal(data, columns, size_train_df, size_criminal_train_df, size_not_criminal_train_df, criminal_dic, not_criminal_dic):
	prob = log10(size_criminal_train_df/size_train_df)
	for i, x in enumerate(columns):
		if(data[i] in criminal_dic.get(x).keys()):
			prob = prob + log10(criminal_dic.get(x).get(data[i])/size_train_df)
		else:
			prob = prob + log10(1/(size_train_df*len(columns)))
	return prob


def probabilty_not_criminal(data, columns, size_train_df, size_criminal_train_df, size_not_criminal_train_df, criminal_dic, not_criminal_dic):
	prob = log10(size_not_criminal_train_df/size_train_df)
	for i, x in enumerate(columns):
		if(data[i] in not_criminal_dic.get(x).keys()):
			prob = prob + log10(not_criminal_dic.get(x).get(data[i])/size_train_df)
		else:
			prob = prob + log10(1/(len(columns)*size_train_df))
	return prob


def prediction(input_file_path, test_file_path):
	train_df = data_preparation(input_file_path)
	size_train_df = train_df.shape[0]
	size_criminal_train_df = train_df[train_df['Criminal']==1].shape[0]
	size_not_criminal_train_df = size_train_df - size_criminal_train_df
	criminal_dic = criminal_dictionary(input_file_path)
	test_df = data_preparation(test_file_path)
	if('Criminal' in test_df.columns):
		answers = train_df['Criminal']
		test_df.drop('Criminal',axis=1,inplace=True)
	not_criminal_dic = non_criminal_dictionary(input_file_path)
	columns = list(test_df.columns)
	columns = columns[1:]
	for i in range(0,10):
		data = list(test_df.loc[i,:])
		data = data[1:]
		prob_criminal = probability_criminal(data, columns, size_train_df, size_criminal_train_df, size_not_criminal_train_df, criminal_dic, not_criminal_dic)
		prob_not_criminal = probabilty_not_criminal(data, columns, size_train_df, size_criminal_train_df, size_not_criminal_train_df, criminal_dic, not_criminal_dic)
		print(prob_criminal, prob_not_criminal)
		print(answers[i])
	return




print(prediction(path.format('/criminal_train.csv'), path.format('/criminal_train.csv')))
