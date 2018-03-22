
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


path = r'/Users/home/Desktop{}'


# uninformative columns to delete found by using feature engineering.
def remove_uniformative_columns(input_file_path):
	train_input_df = pd.read_csv(input_file_path)
	# removed cellwrkng, othinsm, 'iihh65_2',  column from this
	'''columns_to_remove = ['irki17_2','iihh65_2', 'iimedicr', 'iimcdchp', 'iichmpus', 'prvhltin', 'grphltin',
					'hltinnos', 'iiothhlt', 'irinsur4', 'hllosrsn', 'hlnvoffr', 'hlnvcost', 'hlnvneed', 'hlnvsor',
					'anyhlti2', 'govtprog', 'irwelmos', 'irfamin3', 'troubund', 'coutyp2', 'aiind102', 'prxydata',
					'cellwrkng', 'othins']
	columns_to_remove = [i.upper() for i in columns_to_remove]
	train_input_df.drop(columns_to_remove, axis=1, inplace=True)'''
	return train_input_df

# Method for cleaning the data, given the input file name.
def data_preparation_for_building_model(input_file_path):
	train_input_df = remove_uniformative_columns(input_file_path)
	train_input_df.drop(['Criminal'], axis=1, inplace=True)
	train_y = pd.read_csv(input_file_path, usecols=['PERID','Criminal'])
	return train_input_df, train_y

# Method for splitting the input data into training and testing data.
def splitting_cross_validation_set(input_file_path):
	train_x, train_y = data_preparation_for_building_model(input_file_path)
	train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.3)
	return train_x, test_x, train_y, test_y

# Using sklearn built in Logistic Regression Method.
def sklearn_model_accuracy(input_file_path):
	train_x, test_x, train_y, test_y = splitting_cross_validation_set(input_file_path)
	train_y.drop('PERID', axis=1, inplace=True)
	logreg = LogisticRegression(penalty='l1', tol=0.0001, C=1, solver='liblinear', max_iter=500)
	logreg.fit(train_x, train_y)
	predicted_y = logreg.predict(test_x)
	score = accuracy_score(test_y['Criminal'], predicted_y)
	return score


# Predicting for the new testing data using sklearn model.
def sklearn_model_predict(input_file_path, test_file_path):
	# First building model on the input data.
	train_x, train_y = data_preparation_for_building_model(input_file_path)
	logreg = LogisticRegression(penalty='l1', tol=0.0001, C=1, solver='liblinear', max_iter=500)
	logreg.fit(train_x, train_y['Criminal'])
	# Predicting for the test data.
	test_df = remove_uniformative_columns(test_file_path)
	predicted_values = logreg.predict(test_df)
	# Returning the output dataframe.
	final_output_df = pd.DataFrame({})
	final_output_df.insert(0, 'PERID', test_df['PERID'])
	final_output_df.insert(1, 'Criminal', predicted_values)
	return final_output_df


output_df = sklearn_model_predict(path.format('/criminal_train.csv'), path.format('/criminal_test.csv'))
output_df.to_csv(path.format('/submit_solution_sklearn.csv'), index=False)


#print(sklearn_model_accuracy(path.format('/criminal_train.csv')))
