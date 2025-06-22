function [learnableLayer, classLayer] = findLayersToReplace(lgraph)
    %FINDLAYERSTOREPLACE Find layers to replace for transfer learning
    %   [learnableLayer,classLayer] = findLayersToReplace(lgraph) returns the
    %   learnable layer and classification layer of the layer graph lgraph.
    
    layers = lgraph.Layers;
    
    % Find classification layer
    isClassLayer = arrayfun(@(x) isa(x, 'nnet.cnn.layer.ClassificationOutputLayer') || ...
                             isa(x, 'nnet.cnn.layer.SoftmaxLayer'), layers);
    classLayer = layers(find(isClassLayer, 1));
    
    % Find last learnable layer
    isLearnable = arrayfun(@(x) isa(x, 'nnet.cnn.layer.FullyConnectedLayer') || ...
                            isa(x, 'nnet.cnn.layer.Convolution2DLayer'), layers);
    learnableLayer = layers(find(isLearnable, 1, 'last'));
end