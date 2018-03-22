
import pandas as pd
import numpy as np


# Function for computing the correlation coeffiecient between the features.
def correlation_coefficient(feature_1, feature_2):
	feature_1, feature_2 = np.array(feature_1), np.array(feature_2)

	#computing the mean and standard deviation of both the features
	feature_1_mean, feature_2_mean = np.mean(feature_1), np.mean(feature_2)
	feature_1_std, feature_2_std = np.std(feature_1), np.std(feature_2)

	# computing the covarinace of the features.
	covarinace = 0
	for i in range(0,feature_1.size):
		covarinace = covarinace + (feature_1[i] - feature_1_mean)*(feature_2[i] - feature_2_mean)
	covarinace/=feature_1.size

	#returning the final value
	return covarinace/(feature_2_std*feature_1_std)


# Computing the correlation between all the feature, forming a feature matrix.
def correlation_matrix_of_dataframe_features():
	correlation_matrix = np.ones(4900).reshape(70,70)
	for column_iter in range(0,70):
		for row_iter in range(0,70):
			if(correlation_matrix[row_iter][column_iter]!=1):
				continue
			else:
				correlation_matrix[row_iter][column_iter] = correlation_coefficient(train_df_input[train_column_name[row_iter]], train_df_input[train_column_name[column_iter]])
				correlation_matrix[column_iter][row_iter] = correlation_matrix[row_iter][column_iter]
		print(column_iter)
		print(np.count_nonzero(correlation_matrix==1))
	return correlation_matrix

# Path where all the files are located.
path = "/Users/home/Desktop{}"


# train_df_input is the input for making model.
train_df_input = pd.read_csv(path.format('/criminal_train.csv'))
train_df_input.drop(['Criminal'], axis=1, inplace=True)

# train_df_output is the output of training data.
train_df_output = pd.read_csv(path.format('/criminal_train.csv'), usecols=['PERID', 'Criminal'])

# testing data.
test_df = pd.read_csv(path.format('/criminal_test.csv'))

# sample submission data.
sample_submit_df = pd.read_csv(path.format('/sample_submission.csv'))


# Names of columns of training data except PERID column.
train_column_name = list(train_df_input.columns.values)
train_column_name.remove('PERID')

# Forming the correlation matrix of the dataframe, using the function.
correlation_matrix = correlation_matrix_of_dataframe_features()
correlation_matrix_df = pd.DataFrame(correlation_matrix)

# Storing the matrix in dataframe form by function to .csv file.
correlation_matrix_df.columns = train_column_name
correlation_matrix_df.to_csv(path.format('/correlation_data.csv'), index=False)
