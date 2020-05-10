which Lensing_TrainingImage_Generator
load('GREAT_IMS6.mat','GREAT_IMS')
len = size(GREAT_IMS,2)
Lensing_TrainingImage_Generator(len, 5, 1 , 'train')