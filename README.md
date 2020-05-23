# pokemon_GAN

dataset was download from https://www.kaggle.com/vishalsubbiah/pokemon-images-and-types

This GAN work is a test for check the impact of ( using DANN method / using inline param.requires_grad = False ) in the training of GAN

DANN fail
'Teacher Wang YuChang had answer to me, due to DANN cannot train discriminator well so cannot using DANN strategy in GAN'
'I have another view point to DANN, due to the reverse grading method, the reverse way to 0 is not to 1, due to the high dimension space, it's work will diverge from 0 rather than converge to 1'

inline .requires_grad success
