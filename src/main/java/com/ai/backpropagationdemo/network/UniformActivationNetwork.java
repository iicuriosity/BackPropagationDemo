package com.ai.backpropagationdemo.network;

import com.ai.backpropagationdemo.activation.ActivationFunction;
import com.ai.backpropagationdemo.batch.Batch;
import com.ai.backpropagationdemo.layer.Layer;
import com.ai.backpropagationdemo.loss.LossFunction;

import java.util.ArrayList;

public class UniformActivationNetwork {

    private ActivationFunction activationFunction;
    private int numInputs;
    private int numberOfLayers;
    private int[] perceptronsAtEachLayer;
    private LossFunction lossFunction;
    private ArrayList<Layer> layers = new ArrayList<>();
    private double learningRate;

    public UniformActivationNetwork(int numInputs, int numberOfLayers, int[]  perceptronsAtEachLayer,ActivationFunction activationFunction, LossFunction lossFunction, double learningRate) {

        if(numberOfLayers!=perceptronsAtEachLayer.length){
            throw new IllegalArgumentException("Number of layers must be equal to number of perceptrons each layer");
        }

        this.numberOfLayers = numberOfLayers;
        this.perceptronsAtEachLayer = perceptronsAtEachLayer;
        this.lossFunction = lossFunction;
        this.activationFunction = activationFunction;
        this.numInputs = numInputs;
        this.learningRate = learningRate;

        for(int i=0; i<perceptronsAtEachLayer.length; i++){
            int layerInputDimension = i==0? numInputs: perceptronsAtEachLayer[i-1];
            Layer layer = new Layer(layerInputDimension, perceptronsAtEachLayer[i], activationFunction);
            layers.add(layer);
        }


    }

    public void train(Batch trainingBatch, int numberOfIterations){
        // TODO implement this
    }



}