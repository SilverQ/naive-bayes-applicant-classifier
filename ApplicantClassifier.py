# from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.tokenizer import Tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier
import csv
import numpy as np


# https://github.com/muatik/naive-bayes-classifier


tokenizer = Tokenizer()
# https://kkamikoon.tistory.com/119
# related to the tokenizer positional argument error

ApplicantTrainer = Trainer(Tokenizer)

file_encoding = 'cp1252'

# 6 classes
exam01_train = 'data/nm_kind.csv'
exam01_test = 'data/nm_test2.txt'
exam01_result = 'data/nm_test_result2.csv'

# 5 classes without person
exam02_train = 'data/nm_kind_wo_person.txt'
exam02_test = 'data/nm_test2.txt'
exam02_result = 'data/nm_test_wo_person_result2.csv'

with open(exam02_train, newline='', encoding=file_encoding) as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    for applicant in reader:
        print(applicant['nm'], applicant['kind_code'])
        ApplicantTrainer.train(text=applicant['nm'], className=applicant['kind_code'])

ApplicantClassifier = Classifier(ApplicantTrainer.data, tokenizer)
# classification = ApplicantClassifier.classify("sam univ")
# print(classification)

results = []
# with open('data/nm_test.txt') as testfile:
#     for applicant in testfile:
#         classification = np.array(ApplicantClassifier.classify(applicant))
#         # print(applicant, classification[:, 1], classification[0][0])
#         results.append((applicant, classification[0][0]))
#         print(applicant)
#
# print(results[:3])

with open(exam02_test, newline='', mode='rt') as testfile:
    for line in testfile:
        # print(line.strip())
        classification = np.array(ApplicantClassifier.classify(line.strip()))
        # print(classification[0][0])
        results.append((line.strip(), classification[0][0]))


with open(exam02_result, newline='', mode='w', encoding='utf-8') as write_file:
    writer = csv.writer(write_file, delimiter='\t')
    writer.writerows(results)
#
# with open('data/nm_test_result.txt', 'w') as f:
#     for item in results:
#         print(item)
        # f.write('{}\n'.format(item[0], item[1]))
