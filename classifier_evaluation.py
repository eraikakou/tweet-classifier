from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


class ClassifierEvaluation(object):

	def __init__(self):
		pass

	@staticmethod
	def accuracy_and_f1_score_calculation(correct_labels, predicted_labels):
		accuracy = round(accuracy_score(correct_labels, predicted_labels) * 100.00, 2)
		f1score = f1_score(correct_labels, predicted_labels)

		return {'accuracy': accuracy, 'f1_score': f1score}