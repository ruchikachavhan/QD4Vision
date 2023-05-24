# This is a temporary script to measure the legth of datasets 

from test_datasets import *


# train = FacesInTheWild300W('../TestDatasets/300W', split='train')
# valid = FacesInTheWild300W('../TestDatasets/300W', split='valid')
# test = FacesInTheWild300W('../TestDatasets/300W', split='test')

# train = CelebA('../TestDatasets/CelebA', split='train')
# valid = CelebA('../TestDatasets/CelebA', split='valid')
# test = CelebA('../TestDatasets/CelebA', split='test')

# train = LeedsSportsPose('../TestDatasets/LeedsSportsPose', split='train')
# valid = len(train) - int(0.8*len(train))
# test = LeedsSportsPose('../TestDatasets/LeedsSportsPose', split='test')

# train = ALOI('../TestDatasets/ALOI/png4')
# print(len(train))

train = Causal3DIdent('../TestDatasets/Causal3d', split='train', transform=None)
test = Causal3DIdent('../TestDatasets/Causal3d', split='test', transform=None)
print(len(train), len(test))
