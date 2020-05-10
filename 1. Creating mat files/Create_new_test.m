GREAT_IMS = {};
fileList = dir('C:/Users/fedor/For jupyter/Ensai/data_galaxyzoo/images_test_rev1/');
filenames = {fileList.name};
len = size(filenames,2) +3

for k = 3:len
	% Create an image filename, and read it in to a variable called imageData.
	jpgFileName = strcat('C:/Users/fedor/For jupyter/Ensai/data_galaxyzoo/images_test_rev1/', ...
                filenames{k});
	if exist(jpgFileName, 'file')
		imageData = imread(jpgFileName);
        GREAT_IMS{k} = rgb2gray(imresize(imageData, [400 400]));
	else
		fprintf('File %s does not exist.\n', jpgFileName);
    end
end

save('GREAT_IMS31.mat','GREAT_IMS');