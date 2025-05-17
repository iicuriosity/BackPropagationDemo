package com.ai.backpropagationdemo.network;

import com.ai.backpropagationdemo.activation.ActivationFunction;
import com.ai.backpropagationdemo.engine.NetworkExecutionEngine;
import com.ai.backpropagationdemo.layer.Layer;
import com.ai.backpropagationdemo.loss.LossFunction;
import com.ai.backpropagationdemo.training.TrainingPolicy;
import lombok.Getter;

import java.util.ArrayList;

public class UniformActivationNetwork {

    @Getter
    private final ActivationFunction activationFunction;
    @Getter
    private final int numInputs;
    @Getter
    private final int numberOfLayers;
    @Getter
    private final int[] perceptronsAtEachLayer;

    private final ArrayList<Layer> layers = new ArrayList<>();



    private final NetworkExecutionEngine networkExecutionEngine;


    public UniformActivationNetwork(int numInputs, int numberOfLayers, int[]  perceptronsAtEachLayer,ActivationFunction activationFunction) {

        if(numberOfLayers!=perceptronsAtEachLayer.length){
            throw new IllegalArgumentException("Number of layers must be equal to number of perceptrons each layer");
        }

        this.numberOfLayers = numberOfLayers;
        this.perceptronsAtEachLayer = perceptronsAtEachLayer;
        this.activationFunction = activationFunction;
        this.numInputs = numInputs;

        for(int i=0; i<this.perceptronsAtEachLayer.length; i++){
            int layerInputDimension = i==0? this.numInputs: this.perceptronsAtEachLayer[i-1];
            Layer layer = new Layer(layerInputDimension, this.perceptronsAtEachLayer[i], this.activationFunction);
            layers.add(layer);
        }
        networkExecutionEngine = new NetworkExecutionEngine(layers);


    }



    public void train(TrainingPolicy trainingPolicy) {
        // TODO implement this
        trainingPolicy.train(layers);

    }

    public double[] forward(double[] inputs) {
        if (inputs.length != this.numInputs) {
            throw new IllegalArgumentException("Number of inputs must be equal to number of perceptrons each layer");
        }
        return networkExecutionEngine.forward(inputs);
    }

}