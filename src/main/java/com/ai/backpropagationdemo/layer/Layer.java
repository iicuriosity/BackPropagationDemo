package com.ai.backpropagationdemo.layer;

import com.ai.backpropagationdemo.activation.ActivationFunction;
import com.ai.backpropagationdemo.batch.Batch;
import lombok.Getter;

import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.tuple.Pair;


public class Layer {

    private final ActivationFunction activationFunction;

    @Getter
    private double[][] weights;

    @Getter
    private double[] biases;

    @Getter
    private final int inputDimension;

    private final int outputDimension;


    public Layer(int inputDimension, int outputDimension, ActivationFunction activationFunction) {
        this.inputDimension = inputDimension;
        this.outputDimension = outputDimension;
        this.activationFunction = activationFunction;
        initializeWeights();
        initializeBiases();
    }

    public double[] getWeightedInput(double[] input) {
        if(input.length != inputDimension)
            throw new IllegalArgumentException(String.format("The given input's dimension [%s] does not match the layer's input dimension [%s]", input.length,inputDimension));
        double[] weightedInput = new double[outputDimension];
        // computing the weight matrix multiplication = bias
        for(int i = 0; i < outputDimension; i++){
            weightedInput[i] = biases[i];
            for(int j = 0; j < inputDimension; j++){
                weightedInput[i] += weights[i][j]*input[j];
            }
        }
        return weightedInput;
    }

    public Pair<double[],double[]> forward(double[] input) {

        double[] weightedInput = getWeightedInput(input);
        double[] output = activationFunction.apply(weightedInput);

        return  Pair.of(weightedInput,output);
    }

    public double[] derivative(double[]  weightedInput) {
        if(weightedInput.length != outputDimension)
            throw new IllegalArgumentException("The given weighted input's dimension does not match the layer's output dimension");
        return activationFunction.derivative(weightedInput);
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

    public void adjust(double[][] accumulatedDW, double[] accumulatedDB) {
        if(accumulatedDW.length != outputDimension || accumulatedDW[0].length != inputDimension)
            throw new IllegalArgumentException("The given accumulatedDW's dimensions do not match the layer's dimensions");
        if(accumulatedDB.length != outputDimension)
            throw new IllegalArgumentException("The given accumulatedDB's dimensions do not match the layer's dimensions");
        for(int i = 0; i < outputDimension; i++){
            for(int j = 0; j < inputDimension; j++){
                weights[i][j] += accumulatedDW[i][j];
            }
            biases[i] += accumulatedDB[i];
        }
    }
}
