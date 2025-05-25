package com.ai.backpropagationdemo.engine;

import com.ai.backpropagationdemo.layer.Layer;
import com.ai.backpropagationdemo.training.TrainingStrategy;
import lombok.RequiredArgsConstructor;

import java.util.ArrayList;

@RequiredArgsConstructor
public class NetworkExecutionEngine {

    private final ArrayList<Layer> layers;
    private final TrainingStrategy trainingStrategy;

    public double[] forward(double[] inputs) {
        double[] layerInput = inputs;

        for (Layer layer : layers) {
            layerInput = layer.forward(layerInput).getRight();
        }

        return layerInput;
    }

    public void train(){
        trainingStrategy.train(layers);
    }
}
