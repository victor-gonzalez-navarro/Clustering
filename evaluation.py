def evaluate(labels_method, groundtruth_labels):
    n_instances = len(groundtruth_labels)
    f00 = 0; f01 = 0; f10 = 0; f11 = 0;
    for i in range(0, n_instances):
        for j in range(i + 1, n_instances):
            # Different class, different cluster
            if (groundtruth_labels[i] != groundtruth_labels[j]) and (labels_method[i] != labels_method[j]):
                f00 += 1
            # Different class, same cluster
            elif (groundtruth_labels[i] != groundtruth_labels[j]) and (labels_method[i] == labels_method[j]):
                f01 += + 1
            # Same class, different cluster
            elif (groundtruth_labels[i] == groundtruth_labels[j]) and (labels_method[i] != labels_method[j]):
                f10 += + 1
            # Same class, same cluster
            elif (groundtruth_labels[i] == groundtruth_labels[j]) and (labels_method[i] == labels_method[j]):
                f11 += + 1

    score_randstatistic = (f00 + f11) / (f00 + f01 + f10 + f11)
    score_jaccardcoefficient = f11 / (f01 + f10 + f11)
    print('The Rand Statistic score is: ' + str(score_randstatistic))
    print('The Jaccard Coefficient score is: ' + str(score_jaccardcoefficient))