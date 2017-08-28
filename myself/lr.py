# -*- coding: utf-8 -*-

"""
自动微分应用于LR
和soft_max
"""
import codecs

def getLabelAndFeature(path="E:\\CloudMusicProject\\autoDiff\\myself\\x"):
    labels = []
    features = []
    with codecs.open(path) as f:
        for line in f:
            lines = line.strip().split("\t")
            label = int(lines[0])
            feature = [int(x) for x in lines[1]]
            if label < 2:
                labels.append(label)
                features.append(feature)
    return labels, features

###################
#      LR         #
###################




###################
#    softMax      #
###################




if __name__ == "__main__":
    getLabelAndFeature()
    pass


