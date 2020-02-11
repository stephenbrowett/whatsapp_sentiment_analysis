import random

from textblob import TextBlob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

filename = "data/test/WhatsAppChatWithTest.txt"

with open(filename, "r", encoding = "latin1") as infile:
	data = infile.read()

data = data.split('\n')

conversation = defaultdict(lambda : [])
for i in data:
	try:
		statement = i.split(': ')[1]
		person = i.split(': ')[0].split(' - ')[1]
		date = i.split(': ')[0].split(', ')[0]
		time = i.split(': ')[0].split(', ')[1]
	except IndexError:
		pass
	
	blob = TextBlob(statement)

	sentiment = []
	for sentence in blob.sentences:
		sentiment.append(sentence.sentiment.polarity)

	sentiment = np.mean(sentiment)

	conversation[person].append({
		'sentence' : sentence,
		'sentiment' : sentiment,
		'date' : date,
		'time' : time
	})

fig = plt.figure()

f, (ax1, ax2, ax3) = plt.subplots(3)
#ax1 = fig.add_subplot(111)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
plt.gcf().autofmt_xdate(rotation = 0)

ax1.xaxis_date()

hist = []
for n, i in enumerate(conversation.keys()):
	print("Person {0}/{1}:".format(n + 1, len(conversation.keys())))
	print(i)
	print('\n')
	print("Number of messages sent: {0}".format(len(conversation[i])))
	sentiments = [j['sentiment'] for j in conversation[i]]
	print("Mean sentiment score: {0}".format(np.mean(sentiments)))
	print("Median sentiment score: {0}".format(np.percentile(sentiments, 50)))
	print("IQR sentiment score: {0}".format(np.percentile(sentiments, [25, 75])))
	print("Most positive messages:")
	print(np.array([j['sentence'] for j in np.array(conversation[i])[np.where(np.array(sentiments) >= 0.9)[0]]]))
	print('\n')
	print("Most negative messages:")
	print(np.array([j['sentence'] for j in np.array(conversation[i])[np.where(np.array(sentiments) <= -0.9)[0]]]))
	print('\n')

	# determine the confidence region around the median
	trial_mean = []
	for j in range(10000):
		subset = [random.sample(sentiments, 1) for n in range(len(sentiments))]
		subset_mean = np.mean(subset)
		trial_mean.append(subset_mean)

	onesig = np.percentile(trial_mean, [16, 50, 84])
	twosig = np.percentile(trial_mean, [2.5, 50, 97.5])
	threesig = np.percentile(trial_mean, [0.5, 50, 99.5])

	print("One sigma confidence interval: {0}".format(onesig))
	print("Two sigma confidence interval: {0}".format(twosig))
	print("Three sigma confidence interval: {0}".format(threesig))

	print('\n')

	dates = set([j['date'] for j in conversation[i]])

	daily_sentiment_low = []
	daily_sentiment = []
	daily_sentiment_high = []
	for j in dates:
		date_subset = [k['sentiment'] for k in conversation[i] if k == j]

		rang = np.percentile(date_subset, [16, 50, 84])

		daily_sentiment_low.append(rang[0])
		daily_sentiment.append(rang[1])
		daily_sentiment_high.append(rang[2])

	print("Happiest day: {0}".format(dates[np.argmax(daily_sentiment)]))
	print("Saddest day: {0}".format(dates[np.argmin(daily_sentiment)]))

	print('\n')

	ax1.plot(dates, daily_sentiment)
	ax1.fill_between(dates, daily_sentiment_low, daily_sentiment_high, alpha = 0.5)

	bins = np.arange(-1.0, 1.2, 0.1)

	hist.append(np.histogram(sentiments, bins = bins))

	#ax1.hist(sentiments, bins = bins, alpha = 0.5)

plt.show()

