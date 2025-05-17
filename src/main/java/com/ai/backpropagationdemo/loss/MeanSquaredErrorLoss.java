package com.ai.backpropagationdemo.loss;

public class MeanSquaredErrorLoss implements LossFunction {


    @Override
    public double compute(double[] result, double[] expectedValue) {

        if (expectedValue.length != result.length) {
            throw new IllegalArgumentException("Expected array length does not match actual array length");
        }

        double sum = 0;
        for (int i = 0; i < result.length; i++) {
         sum += Math.pow(expectedValue[i] - result[i], 2);
        }

        return (sum)/2;
    }

    @Override
    public double[] computeDerivative(double[] result, double[] expectedValue) {

        if (expectedValue.length != result.length) {
            throw new IllegalArgumentException("Expected array length does not match actual array length");
        }

        double[] gradient = new double[result.length];

        for (int i = 0; i < result.length; i++) {
            gradient[i] = (expectedValue[i] - result[i]);
        }

        return gradient;
    }


}
