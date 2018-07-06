from collections import Counter
from random import randrange


class BaselineClassifier(object):

	MAJORITY_CLF = "majority"
	RANDOM_CLF = "random"
	SEED = 123456

	def __init__(self, train_instances, classifier="majority"):
		self.classifier = classifier
		self.train_instances = train_instances
		self.most_frequent_class = None
		self.classes_list = None
		self.unique_classes = None

	def train(self, num_of_train):
		self.classes_list = [item[0] for item in self.train_instances[:num_of_train]]
		self.unique_classes = list(set(self.classes_list))

		if self.classifier == self.MAJORITY_CLF:
			classes_frequency = Counter(self.classes_list).most_common()
			self.most_frequent_class = classes_frequency[0][0]

	def predict(self, unseen_data):
		correct_classes = [int(item[0]) for item in unseen_data]
		predicted_classes = None

		if self.classifier == self.MAJORITY_CLF:
			predicted_classes = [int(self.most_frequent_class)] * len(correct_classes)

		if self.classifier == self.RANDOM_CLF:
			predicted_classes = list()
			for tweet in unseen_data:
				index = randrange(len(self.unique_classes))
				predicted_classes.append(int(self.unique_classes[index]))

		return {'correct_classes': correct_classes, 'predicted_classes' : predicted_classes}
