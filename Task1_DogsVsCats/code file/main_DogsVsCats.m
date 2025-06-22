function main_DogsVsCats()
% MAIN_DOGSVSCATS - Dogs vs Cats classification with Grad-CAM explanation
% This script performs transfer learning on GoogLeNet for binary classification
% and generates Grad-CAM visualizations for test images.

%% Step 1: Load and Prepare Data
disp('Step 1: Loading and preparing data...');
imds = imageDatastore('PetImages', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames'); 
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8);

%% Step 2: Load Pretrained Network
disp('Step 2: Loading GoogLeNet...');
net1 = googlenet;
lgraph = layerGraph(net1);

%% Step 3: Modify Network Architecture
disp('Step 3: Modifying network architecture...');
% Replace classification layers
numClasses = numel(categories(imdsTrain.Labels));
newFCLayer = fullyConnectedLayer(numClasses, ...
    'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);
lgraph = replaceLayer(lgraph, 'loss3-classifier', newFCLayer);

newClassLayer = classificationLayer('Name', 'new_classoutput');
lgraph = replaceLayer(lgraph, 'output', newClassLayer);

% Freeze initial layers
layers = lgraph.Layers;
connections = lgraph.Connections;
for i = 1:10
    if isprop(layers(i), 'WeightLearnRateFactor')
        layers(i).WeightLearnRateFactor = 0;
        layers(i).BiasLearnRateFactor = 0;
    end
end
lgraph = createLgraphUsingConnections(layers, connections);

%% Step 4: Data Augmentation
disp('Step 4: Setting up data augmentation...');
inputSize = net1.Layers(1).InputSize;
pixelRange = [-30 30];
scaleRange = [0.9 1.1];

imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange, ...
    'RandXScale', scaleRange, ...
    'RandYScale', scaleRange);

augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

%% Step 5: Train Network
disp('Step 5: Training network...');
miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);

options = trainingOptions('sgdm', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 6, ...
    'InitialLearnRate', 3e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', valFrequency, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net1 = trainNetwork(augimdsTrain, lgraph, options);

%% Step 6-7: Test Classification with Grad-CAM
disp('Step 6-7: Running classification and Grad-CAM...');
imageFiles = ["amber.jpg", "snoopy.jpg", "sherlock.jpg"];

for i = 1:length(imageFiles)
    % Read and resize image
    img = imread(imageFiles(i));
    img = imresize(img, inputSize(1:2));
    
    % Classify image
    [YPred, scores] = classify(net1, img);
    
    % Display original image with prediction
    figure;
    imshow(img);
    title(sprintf('Original: %s\nPredicted: %s (%.2f%%)', ...
         imageFiles(i), string(YPred), max(scores)*100));
    
    % Generate and display Grad-CAM
    gradcamMap = gradCAM(net1, img, YPred);
    
    figure;
    imshow(img);
    hold on;
    imagesc(gradcamMap, 'AlphaData', 0.5);
    colormap jet;
    colorbar;
    title(sprintf('Grad-CAM: %s\nHighlighted Regions', string(YPred)));
    hold off;
end

%% Step 8: Validation Accuracy
disp('Step 8: Calculating validation accuracy...');
[YPredValidation, ~] = classify(net1, augimdsValidation);
accuracy = sum(YPredValidation == imdsValidation.Labels)/numel(imdsValidation.Labels);
disp(['Validation Accuracy: ', num2str(accuracy*100), '%']);

disp('All steps completed!');
end