
from __future__ import division
import pandas as pd
import numpy as np
import operator
from collections import Counter
from math import log10, fabs
from sklearn.naive_bayes import BernoulliNB

path = '/Users/home/Desktop{}'

'''
Output list of features according to percent of information.
'''

def correlation_data(correlation_data_file_path, percent_information):
	input_df = pd.read_csv(correlation_data_file_path)
	eigen_values, eigen_vectors = np.linalg.eig(np.array(input_df))
	output_dic = zip(list(input_df.columns), eigen_values)
	output_dic = sorted(output_dic, key=operator.itemgetter(1), reverse=True)
	eigen_values = sorted(eigen_values, reverse=True)
	sum_eigen_values = sum(eigen_values)
	feature_engineered_columns = []
	for i,x in enumerate(output_dic):
		if((sum(eigen_values[0:i+1])/sum_eigen_values)*100>float(percent_information)):
			break
		else:
			feature_engineered_columns.append(x[0])
	return feature_engineered_columns

''''
Transform the input data so that model can be built upon that.
'''
def data_preparation(input_file_path, correlation_data_file_path, percent_information):
	input_df = pd.read_csv(input_file_path)
	perid_column = input_df['PERID']
	output_column = []
	selected_columns = correlation_data(correlation_data_file_path, percent_information)
	for i in list(input_df.columns):
		if(i in selected_columns):
			continue
		elif(i=='Criminal'):
			output_column = input_df['Criminal']
			input_df.drop('Criminal',axis=1,inplace=True)
		else:
			input_df.drop(i,axis=1,inplace=True)
	input_df.insert(0,'PERID',perid_column)
	return input_df, output_column

'''
Making a dictionary, first value of the key == not_criminal and second value == criminal.
'''
def count_dictionary(input_file_path, correlation_data_file_path, percent_information):
	input_df, output_column = data_preparation(input_file_path, correlation_data_file_path, percent_information)
	input_df['Criminal'] = output_column
	columns = list(input_df.columns)
	columns.remove('PERID')
	output_dic = {}
	for i in columns:
		# For non criminal persons.
		value_not_criminal = dict(Counter(input_df[i][input_df['Criminal']==0]))
		for j in value_not_criminal.keys():
			value_not_criminal[j] = [value_not_criminal[j],0]
		output_dic[i] = value_not_criminal
		# For criminal persons .
		value_criminal = dict(Counter(input_df[i][input_df['Criminal']==1]))
		for j in value_criminal.keys():
			if(j in output_dic[i].keys()):
				output_dic[i][j][1] = value_criminal[j]
			else:
				output_dic[i][j] = [0,value_criminal[j]]
	return output_dic

'''
Will predict prob of criminal and not criminal given an input data row.
'''
def predict_probabilty(output_dic, count_criminal, count_not_criminal, count_total, number_features_dataframe, features, data):
	prob_criminal = fabs(log10(count_criminal/count_total))
	prob_not_criminal = fabs(log10(count_not_criminal)/count_total)
	for i,x in enumerate(features):
		if(x=='PERID'):
			continue
		else:
			prob_criminal = prob_criminal + fabs(log10(output_dic[x][data[i]][1] + 1/count_criminal))
			prob_not_criminal = prob_not_criminal + fabs(log10(output_dic[x][data[i]][0] + 1/count_not_criminal))
	print(prob_criminal, prob_not_criminal)
	return

'''
Prediction Model, will predict for my complete file .
'''
def prediction_model(input_file_path, test_file_path, correlation_data_file_path, percent_information):
	output_dic = count_dictionary(input_file_path, correlation_data_file_path, percent_information)
	test_df, output_column = data_preparation(test_file_path, correlation_data_file_path, percent_information)
	features_test_df = list(test_df.columns)
	input_df, output_column = data_preparation(input_file_path, correlation_data_file_path, percent_information)
	input_df['Criminal'] = output_column
	count_criminal = input_df[input_df['Criminal']==1].shape[0]
	count_not_criminal = input_df[input_df['Criminal']==0].shape[0]
	count_total = input_df.shape[0]
	number_features = input_df.shape[1] - 1
	for i in range(0,10):
		data = test_df.loc[i,:]
		print(predict_probabilty(output_dic, count_criminal, count_not_criminal, count_total, number_features, features_test_df, data))
		print(input_df.loc[i,'Criminal'])
		print('\n\n')
	return

'''
sklearn Naive Bayes Model
'''

def prediction(input_file_path, test_file_path, correlation_data_file_path, percent_information):
	input_df, output_column = data_preparation(input_file_path, correlation_data_file_path, percent_information)
	clf = BernoulliNB()
	clf.fit(input_df, output_column)
	test_df, output_col = data_preparation(test_file_path, correlation_data_file_path, percent_information)
	out_df = pd.DataFrame({})
	out_df['PERID'] = test_df['PERID']
	out_df['Criminal'] = clf.predict(test_df)
	return out_df

df = prediction(path.format('/criminal_train.csv'), path.format('/criminal_test.csv'), path.format('/correlation_data.csv'), 98)
df.to_csv(path.format('/naive_bayes.csv'), index=False)
