package com.ai.backpropagationdemo.layer;

import com.ai.backpropagationdemo.activation.ActivationFunction;
import lombok.Getter;

public class Layer {

    private ActivationFunction activationFunction;

    private double[][] weights;

    private double[] biases;

    private final int inputDimension;

    private final int outputDimension;

    private double[] weightedInput;

    @Getter
    private double[] output;

    @Getter
    private double[] signal;

    public Layer(int inputDimension, int outputDimension, ActivationFunction activationFunction) {
        this.inputDimension = inputDimension;
        this.outputDimension = outputDimension;
        this.activationFunction = activationFunction;
        initializeWeights();
        initializeBiases();


    }

    public double[] forward(double[] input) {
        if(input.length != inputDimension)
            throw new IllegalArgumentException("The given input's dimension does not match the layer's input dimension");

        output = new double[outputDimension];
        weightedInput = new double[outputDimension];
        // computing the weight matrix multiplication = bias
        for(int i = 0; i < outputDimension; i++){
            weightedInput[i] = biases[i];
            for(int j = 0; j < inputDimension; j++){
                weightedInput[i] += weights[i][j]*input[j];
            }
        }

        output = activationFunction.apply(weightedInput);
        return output;
    }

    public double[] derivative() {
        return activationFunction.derivative(weightedInput);
    }

    public void setSignal(double[] signal) {
        if(signal.length != outputDimension)
            throw new IllegalArgumentException("The given signal's dimension does not match the layer's output dimension");
        this.signal = signal;
    }

    private void initializeWeights() {
        this.weights = new double[outputDimension][inputDimension];
        for (int i = 0; i < outputDimension; i++) {
            for (int j = 0; j < inputDimension; j++) {
                weights[i][j] = Math.random();
            }
        }
    }

    private void initializeBiases() {
        this.biases = new double[outputDimension];
        for (int i = 0; i < outputDimension; i++) {
            biases[i] = Math.random();
        }
    }

}
