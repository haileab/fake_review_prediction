import numpy as np

##################################################
# Function: reviewerData
# input: meta file name, reviewer file name
# output:
# 	Dictionary of Reviewer features:
# 	key : reviewerID
# 	Value: a list that contains
# 	1. maximum number of reviews written in a day
# 	2. ratio of positive reviews (rating = 4/5)
# 	3. ratio of negative reviews (rating = 1/2)
# 	4. average review length
# 	5. standard deviation of his/her all ratings
def reviewerData(meta_file, review_file):
	dataLabel = []
	fmeta = open(meta_file, 'r')
	freviews = open(review_file, 'r')
	meta = fmeta.readlines()
	reviews = freviews.readlines()
	label = {"N": 1, "Y": 0}
	data = {}
	for i in range(len(meta)):
		info = meta[i].split(' ')
		if(not (info[2] in data)):
			data[info[2]] = []
		data[info[2]].append([info[0], info[1], info[3], label[info[4]], int(info[8]), len(reviews[i])])
		dataLabel.append(label[info[4]])
		# data[reviewerID] = [Date,  revieweID, productID, Label,        star,        len(review)]
	#fmeta.close()
	#freviews.close()

	infoExtract = {}
	for key in data:
		date = [x[0] for x in data[key]]
		sorted(date)
		maxLen = 0
		cur = 1
		#print len(date)
		for i in range(1, len(date)):
			if(date[i] != date[i - 1]):
				maxLen = max(maxLen, cur)
				cur = 1
			else:
				cur += 1
		maxLen = max(maxLen, cur)

		rating = np.array([x[4] for x in data[key]])
		reviewLength = np.array([x[5] for x in data[key]])
		infoExtract[key] = [maxLen, sum(rating >= 4) * 1.0 / len(rating), sum(rating <= 2) * 1.0 / len(rating), np.mean(reviewLength), np.std(rating)]




	# pos_neg = np.load('pos_neg.npy')
	# m,n = pos_neg.shape

	review = np.load('rev.dat.npy')
	m, n2 = review.shape

	bigram = np.load('topic_dists_bigrams.npy')
	m, nbi = bigram.shape

	unigram = np.load('topic_dists_unigrams.npy')
	m, nuni = unigram.shape

	dataLabel = np.array(dataLabel)
	dataLabel = dataLabel[0:m]
	np.save('label', dataLabel)

	reviewer = np.zeros((m, 5))
	combo = np.zeros((m,  n2 +5 + nbi + nbi))
	for i in range(m):
		info = meta[i].split(' ')
		reviewer[i] = infoExtract[info[2]]

	# mm = 5
	# combo[:, 0:mm] = reviewer
	# combo[:, mm : mm + n2] = review
	# combo[:, mm + n2 : mm + n2 + nbi] = bigram
	# combo[:, mm + n2 + nbi : mm + n2 + nbi + nuni] = unigram
	# np.save('feature_combo', combo)


	fmeta.close()
	freviews.close()

	return infoExtract


def main():
	reviewerFeature = reviewerData('Data/YelpChi/output_meta_yelpHotelData_NRYRcleaned.txt', 'Data/YelpChi/output_review_yelpHotelData_NRYRcleaned.txt')
	#reviewerFeature = reviewerData('meta_myfile.txt', 'review_myfile.txt')

	# f = np.load('feature_combo.npy')
	# print f.shape
	# 
	#
	# p = np.load('label.npy')
	# print p.shape
	# print p

	#print f[30]


if __name__ == '__main__':
    main()
