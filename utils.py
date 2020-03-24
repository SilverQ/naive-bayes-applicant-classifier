with open('data/nm_test.txt') as testfile:
    for line in testfile:
        print(line)
        # classification = ApplicantClassifier.classify(applicant)
        # test.append(applicant, classification)
#
# print(test[:10])
#
# with open('data/nm_test_result.csv', 'w', encoding=file_encoding) as write_file:
#     writer = csv.writer(write_file)