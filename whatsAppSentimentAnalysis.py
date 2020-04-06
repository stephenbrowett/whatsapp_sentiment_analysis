import random
import datetime

from textblob import TextBlob
from dateutil.parser import parse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

filename = "data/solveig/WhatsAppChatWithSolveigAndvig.txt"

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

# TODO: plot sentiment vs time

fig = plt.figure()

f, (ax1, ax2, ax3, ax4) = plt.subplots(4)
#ax1 = fig.add_subplot(111)

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
#plt.gcf().autofmt_xdate(rotation = 0)

ax1.xaxis_date()

person_sentiment = []
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
	print("")
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

	dates = list(set([datetime.datetime.strptime(j['date'], "%d/%m/%Y") for j in conversation[i]]))

	dates.sort()

	first_date = dates[0]
	last_date = dates[-1]

	all_dates = [first_date + datetime.timedelta(days = i) for i in range((last_date - first_date).days)]

	daily_sentiment_low = []
	daily_sentiment = []
	daily_sentiment_high = []
	num_messages = []
	for j in all_dates:
		date_subset = [k['sentiment'] for k in conversation[i] if k['date'] == j.strftime("%d/%m/%Y")]

		if date_subset == []:
			daily_sentiment_low.append(np.nan)
			daily_sentiment.append(np.nan)
			daily_sentiment_high.append(np.nan)

			num_messages.append(0)

			continue

		rang = np.percentile(date_subset, [16, 50, 84])

		daily_sentiment_low.append(rang[0])
		daily_sentiment.append(rang[1])
		daily_sentiment_high.append(rang[2])

		num_messages.append(len(date_subset))

	print("Happiest day: {0}".format(all_dates[np.nanargmax(daily_sentiment)]))
	print("Saddest day: {0}".format(all_dates[np.nanargmin(daily_sentiment)]))

	print('\n')

	x_data = [date.date() for date in all_dates]

	# ax1 = daily sentiment
	ax1.plot(x_data, daily_sentiment)
	ax1.fill_between(x_data, daily_sentiment_low, daily_sentiment_high, alpha = 0.5)

	# ax2 = number of messages per day

	ax2.plot(x_data, num_messages, label = i)

	# ax3 = histogram of sentiment

	bins = np.arange(-1.0, 1.2, 0.1)

	ax3.hist(sentiments, bins = bins, alpha = 0.5)

	person_sentiment.append(sentiments)

# ax4 = scatter plots of sentiment with best-fit lines and correlation coefficients
people = list(conversation.keys())

ax4.scatter(person_sentiment[0], label = people[0])
ax4.scatter(person_sentiment[1], label = people[1])

plt.legend()

plt.show()

