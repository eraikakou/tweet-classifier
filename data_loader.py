import re
import csv
import re
import html
import random

class DataLoader(object):
	""" """
	path	 = None
	data	 = []

	# <Jenny>
	# Repeating words like yeaaahhhh
	rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE);
	# URLs
	url_regex = re.compile(r"((www\.[a-zA-Z0-9\./]+)|(Www\.[a-zA-Z0-9\./]+)|(WWW\.[a-zA-Z0-9\./]+)|(Http|http|https|ftp)://[a-zA-Z0-9\./]+)")
	# Handles: On Twitter, your username, or handle, is your identity.
	hndl_regex = re.compile(r"@(\w+)")
	# Hashtags
	hash_regex = re.compile(r"#(\w+)")

	# Emoticons
	happy_emoticons = \
		[('__EMOT_SMILEY', [':-)', ':)', '(:', '(-:', ]), \
		 ('__EMOT_LAUGH', [':-D', ':D', 'X-D', 'XD', 'xD', ]), \
		 ('__EMOT_LOVE', ['<3', ':\*', ]), \
		 ('__EMOT_WINK', [';-)', ';)', ';-D', ';D', '(;', '(-;', ]), \
		 ]

	sad_emoticons = \
		[('__EMOT_FROWN', [':-(', ':(', '(:', '(-:', ]), \
		 ('__EMOT_CRY', [':,(', ':\'(', ':"(', ':((']), \
		 ]
	# </Jenny>

	# For reproducible results
	SEED	 = 27231756

	def __init__(self, path):
		self.path = path
		random.seed(self.SEED)

	def read_data_from_csv(self):
		with open(self.path, 'r') as csvfile:
			tweetreader = csv.reader(csvfile, delimiter=',', quotechar='"')

			# Remove header (python csvreader has no built-in functionality to skip it)
			next(tweetreader)

			for tweet in tweetreader:
				# We only need the sentiment and sentiment_text fields
				# self.data.append([tweet[1],self.data_cleaning(tweet[3])])
				self.data.append([tweet[1],tweet[3]])
			# print(self.data)

	def split_dataset(self):
		"""
			Splits the dataset into train, devtest and test.
			We are re-shuffling them and then pick the corresponding slices.
		"""
		random.shuffle(self.data)
		#print("\n\n{0}".format(self.data))
		train	 = self.data[:int(0.8*len(self.data))]
		devtest   = self.data[int(0.8*len(self.data)):int(0.9*len(self.data))]
		test	  = self.data[int(0.9*len(self.data)):]

		return train, devtest, test

	def data_cleaning(self, tweet_text):

		line = re.sub('\.\.+', '.',tweet_text)
		# Count the number of URLs in each tweet
		# numberOfUrls = self.countUrls(line)
		current_tweet = self.process_urls(line)
		# current_tweet = self.processRepeatings(current_tweet)

		# Count the number of mentions in each tweet and replace mentions with a specific token
		# numberOfMentions = self.countHandles(current_tweet)
		# current_tweet = self.processHandles(current_tweet)

		# Count the number of hashtags in each tweet and replace hashtags with a specific token
		# numberOfHashtags = self.countHashtags(current_tweet)
		# current_tweet = self.processHashtags(current_tweet)

		# Remove HTML Character Entities
		# current_tweet = html.unescape(current_tweet)

		# Remove digits
		# current_tweet = self.processDigits(current_tweet)
		# numberOfHappyEmoticons = self.countHappyEmoticons(current_tweet)
		# numberOfSadEmoticons = self.countSadEmoticons(current_tweet)

		return current_tweet

	def rpt_repl(self, match):
		return match.group(1) + match.group(1)

	def processRepeatings(self, tweet_text):
		return re.sub(self.rpt_regex, self.rpt_repl, tweet_text)

	# Remove all digits
	def processDigits(self, tweet_text):
		return re.sub("\d+", "", tweet_text)

	# Remove all URLs in each tweet
	def process_urls(self, tweet_text):
		#return re.sub(self.url_regex, 'URL', tweet_text)
		return re.sub(self.url_regex, '', tweet_text)

	# Count the number of URLs in each tweet
	def countUrls(self, tweet_text):
		return len(re.findall(self.url_regex, tweet_text))

	# Remove all ‘@’ mentions in each tweet //Replace all ‘@’ mentions in each tweet with token USER_MENTION
	def processHandles(self, tweet_text):
		#return re.sub(self.hndl_regex, 'USER_MENTION', tweet_text)
		return re.sub(self.hndl_regex, '', tweet_text)

	# Count the number of ‘@’ mentions in each tweet
	def countHandles(self, tweet_text):
		return len(re.findall(self.hndl_regex, tweet_text))

	# Remove all ‘#' hashtags in each tweet //Replace all ‘#' hashtags in each tweet with token __HASH_
	def processHashtags(self, tweet_text):
		return re.sub(self.hash_regex, self.hash_repl, tweet_text)

	def hash_repl(self, match):
		#return '__HASH_' + match.group(1).upper()
		return '' + match.group(1).upper()

	# Count the number of ‘#' hashtags in each tweet
	def countHashtags(self, tweet_text):
		return len(re.findall(self.hash_regex, tweet_text))

	# For emoticon regexes
	def escape_paren(self, arr):
		return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]

	def regex_union(self, arr):
		return '(' + '|'.join(arr) + ')'

	def countHappyEmoticons(self, tweet_text):
		emoticons_regex = [(repl, re.compile(self.regex_union(self.escape_paren(regx)))) \
						   for (repl, regx) in self.happy_emoticons]
		count = 0
		for (repl, regx) in emoticons_regex :
			count += len( re.findall( regx, tweet_text) )
		return count

	def countSadEmoticons(self, tweet_text):
		emoticons_regex = [(repl, re.compile(self.regex_union(self.escape_paren(regx)))) \
						   for (repl, regx) in self.sad_emoticons]
		count = 0
		for (repl, regx) in emoticons_regex :
			count +=len( re.findall( regx, tweet_text) )
		return count

