from data_loader import DataLoader
from classifier_evaluation import ClassifierEvaluation
from neural_net_classifier import NNClassifier
from baseline_classifier import BaselineClassifier
from random import seed


def main():

	# Training dataset sizes for learning curves generation
	# learning_sizes = [100, 1000, 2000, 5000, 10000, 20000, 50000, 71980]
	# testload = DataLoader("data/sentiment_sample_10K.csv")
	testload = DataLoader("data/sentiment_analysis_dataset.csv")
	testload.read_data_from_csv()

	# Get the data subsets
	train, devtest, test = testload.split_dataset()

	baseline = BaselineClassifier(train, "random")
	seed(baseline.SEED)
	baseline.train(len(train))
	test_predictions = baseline.predict(test)
	clf_evaluation = ClassifierEvaluation()
	scores = clf_evaluation.accuracy_and_f1_score_calculation(test_predictions['correct_classes'], test_predictions['predicted_classes'])
	print('The accuracy of ' + baseline.classifier + ' classifier is ' + str(scores['accuracy']) + '% and the f1 score is ' + str(scores['f1_score']))

	baseline = BaselineClassifier(train, "majority")
	baseline.train(len(train))
	test_predictions = baseline.predict(test)
	clf_evaluation = ClassifierEvaluation()
	scores = clf_evaluation.accuracy_and_f1_score_calculation(test_predictions['correct_classes'], test_predictions['predicted_classes'])
	print('The accuracy of ' + baseline.classifier + ' classifier is ' + str(scores['accuracy']) + '% and the f1 score is ' + str(scores['f1_score']))

	nnclf = NNClassifier()
	nnclf.train(train, devtest)
	# Finally checking on test
	# This was only run once after we have finished trying model architectures
	# And tuning parameters.
	nnclf.predict(test)

if __name__ == '__main__':
	main()