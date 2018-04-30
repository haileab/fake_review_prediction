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
		print(data[info[2]])
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

def main():
	reviewerData('Data/YelpChi/output_meta_yelpHotelData_NRYRcleaned.txt', 'Data/YelpChi/output_review_yelpHotelData_NRYRcleaned.txt')
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
