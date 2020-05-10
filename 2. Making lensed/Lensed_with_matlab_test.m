which Lensing_TrainingImage_Generator
load('GREAT_IMS_test.mat','GREAT_IMS')
len = size(GREAT_IMS,2)
Lensing_TrainingImage_Generator(len, 10, 1 , 'train')