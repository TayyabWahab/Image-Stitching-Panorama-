    %Clear Everything before starting
clc
clear all 
close all

%Load Images
Img1 = imread('Stitch2.jpeg');
Img2 = imread('Stitch1.jpeg');

%Convert to double
img1 = im2double(Img1);
img2 = im2double(Img2);

%Convert to Grayscale images
gray_1 = rgb2gray(img1);
gray_2 = rgb2gray(img2);

%Detect Features in both Images
points1 = detectSURFFeatures(gray_1);
points2 = detectSURFFeatures(gray_2);

%Extract Features of both Images
[features1, points1] = extractFeatures(gray_1, points1); 
[features2, points2] = extractFeatures(gray_2, points2); 


tforms(1) = projective2d(eye(3));
tforms(2) = projective2d(eye(3));
img_size = zeros(2,2);

   
% Get image size.
img_size(1,:) = size(gray_1);
img_size(2,:) = size(gray_2);
       
  
% Find correspondences between two Images
indexPairs = matchFeatures(features2, features1, 'Unique', true);
       
%Save the matched points
matchedPoints1 = points1(indexPairs(:,2), :);
matchedPoints2 = points2(indexPairs(:,1), :);        
    
% Estimate the Geometirc transformation between two Images.
tforms(1) = estimateGeometricTransform(matchedPoints2, matchedPoints1,'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
tforms(2) = estimateGeometricTransform(matchedPoints1, matchedPoints2,'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);

tforms(1).T = tforms(1).T * tforms(2).T; 
tforms(2).T = tforms(2).T * tforms(1).T; 
    
% Compute the boundries for transformation
for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 img_size(i,2)], [1 img_size(i,1)]);    
end 
avgXLim = mean(xlim, 2);
[~, idx] = sort(avgXLim);
center_dx = floor((numel(tforms)+1)/2);
cent_Image = idx(center_dx);
inv_T = invert(tforms(cent_Image));
tforms(1).T = tforms(1).T * inv_T.T;    
tforms(2).T = tforms(2).T * inv_T.T;
for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 img_size(i,2)], [1 img_size(i,1)]);
end
maximg_size = max(img_size);

% Compute the Boundries for Stitched Images
xMin = min([1; xlim(:)]);
xMax = max([maximg_size(2); xlim(:)]);
yMin = min([1; ylim(:)]);
yMax = max([maximg_size(1); ylim(:)]);
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize Image for Panorama
panorama = zeros([height width 3], 'like', Img1);


blender = vision.AlphaBlender('Operation', 'Binary mask','MaskSource', 'Input port');  

xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);
 
   
% Transform images into Panorama
warpedImage1 = imwarp(Img1, tforms(2), 'OutputView', panoramaView);
warpedImage2 = imwarp(Img2, tforms(1), 'OutputView', panoramaView);
                  
% Generate a binary mask.    
mask1 = imwarp(true(size(Img1,1),size(Img1,2)), tforms(2), 'OutputView', panoramaView);
mask2 = imwarp(true(size(Img2,1),size(Img2,2)), tforms(1), 'OutputView', panoramaView);
    
% Stitch Images to make Panorama.
panorama = step(blender, panorama, warpedImage1, mask1);
panorama = step(blender, panorama, warpedImage2, mask2);

p = imcrop(panorama,[0,20,size(panorama,2),size(panorama,1)-20]);
figure 
imshow(Img1)
title('left Image')
figure 
imshow(Img2)
title('Right Image') 
figure
imshow(p)
title('Stitched Image')
    