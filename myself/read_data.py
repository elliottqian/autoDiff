# -*- coding: utf-8 -*-

import codecs

path = "C:\\Users\qianwei\Desktop\机器学习实战及配套代码\MLiA_SourceCode\machinelearninginaction\Ch02\digits\\testDigits\\"


def getLabelFeature(path, space=3):
    import os
    file_names = os.listdir(path)
    i = 1

    features = []
    labels = []

    for filename in file_names:

        labels.append(filename.split("_")[0])

        file_path_one = path + filename
        with codecs.open(file_path_one, encoding="utf-8") as f:
            r = read_a_file(f, space=space)
            features.append(r)

    ff = codecs.open("x", encoding="utf-8", mode="wb")
    for i in range(len(labels)):
        print(labels[i] + "\t" + features[i])
        ff.write(labels[i] + "\t" + features[i] + "\n")
    ff.close()


def read_a_file(f, space=2):
    i = 0
    result = ""
    for line in f:
        line = line.strip()
        i += 1
        if i % space == 0:
            # print(line)
            x = ""
            for j in range(len(line)):
                if j % space == 0:
                    x = x + line[j]
            result = result + x
    return result

if __name__ == "__main__":
    getLabelFeature(path)
    pass
