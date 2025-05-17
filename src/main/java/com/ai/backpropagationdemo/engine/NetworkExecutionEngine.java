package com.ai.backpropagationdemo.engine;

import com.ai.backpropagationdemo.layer.Layer;

import java.util.ArrayList;

public class NetworkExecutionEngine {

    private final ArrayList<Layer> layers;

    public NetworkExecutionEngine(ArrayList<Layer> layers) {
        this.layers = layers;

    }

    public double[] forward(double[] inputs) {
        double[] layerInput = inputs;

        for (Layer layer : layers) {
            layerInput = layer.forward(layerInput).getRight();
        }

        return layerInput;
    }
}
