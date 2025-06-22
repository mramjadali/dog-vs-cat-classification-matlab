function lgraph = createLgraphUsingConnections(layers, connections)
% Create a layer graph using the specified layers and connections

lgraph = layerGraph();

% Add each layer
for i = 1:numel(layers)
    lgraph = addLayers(lgraph, layers(i));
end

% Connect each layer
for i = 1:size(connections,1)
    lgraph = connectLayers(lgraph, ...
        connections.Source{i}, ...
        connections.Destination{i});
end
end
