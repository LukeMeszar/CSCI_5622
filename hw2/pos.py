# import nltk
# from nltk.corpus import brown
#
# suffix_fdist = nltk.FreqDist()
# for word in brown.words():
#     word = word.lower()
#     suffix_fdist[word[-1:]] += 1
#     suffix_fdist[word[-2:]] += 1
#     suffix_fdist[word[-3:]] += 1
#     suffix_fdist[word[-4:]] += 1
#
# common_suffixes = [suffix for (suffix, count) in suffix_fdist.most_common(100)]
# print(common_suffixes)
#
# def pos_features(word):
#     features = {}
#     for suffix in common_suffixes:
#         features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
#     return features
#
# tagged_words = brown.tagged_words(categories='news')
# print("tagged words")
# featuresets = [(pos_features(n), g) for (n,g) in tagged_words]
# print("featuresets")
# size = int(len(featuresets) * 0.1)
# train_set, test_set = featuresets[size:], featuresets[:size]
# print("train_set")
# classifier = nltk.DecisionTreeClassifier.train(train_set)
# print("classifier")
# nltk.classify.accuracy(classifier, test_set)
# classifier.classify(pos_features('cats

suffixes = [',', 'e', '.', 'the', 's', 'of', 'd', 'he', 'a', 't', 'n', 'to', 'in', 'and', 'y', 'r', 'is', 'f', 'o', 'ed', 'on', 'nd', 'as', 'l', 'g', 'at', 'ng', 'er', 'it', 'ing', '``', "''", 'h', 'or', 'es', ';', 're', 'i', 'an', 'was', 'be', 'his', 'for', '?', 'm', 'ly', 'by', 'ion', 'en', 'al', 'nt', 'hat', 'st', 'th', 'tion', 'me', 'll', 'her', 'le', 'ce', 'ts', "'", 'that', 've', 'se', '--', 'had', 'ut', ')', '(', 'are', 'not', 'ent', 'ch', 'k', 'w', 'ld', '`', 'but', 'rs', 'ted', 'one', 'ere', 'ne', 'we', 'all', 'ns', 'ith', 'ad', 'ry', 'with', ':', 'te', 'so', 'out', 'if', 'you', 'no', 'ay', 'ty']
print(suffixes)
