function layers = freezeWeights(layers)
    for i = 1:numel(layers)
        if isprop(layers(i),'WeightLearnRateFactor')
            layers(i).WeightLearnRateFactor = 0;
            layers(i).BiasLearnRateFactor = 0;
        end
    end
end