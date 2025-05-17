package com.ai.backpropagationdemo.activation;

public class SigmoidActivation implements ActivationFunction {

    @Override
    public double[] apply(double[] z) {
        double [] output = new double[z.length];

        for (int i = 0; i < z.length; i++) {
            output[i] = 1/(1+Math.exp(-z[i]));
        }

        return output;
    }

    @Override
    public double[] derivative(double[] z) {
        double [] output = new double[z.length];

        for (int i = 0; i < z.length; i++) {
            output[i] = Math.exp(-z[i])/Math.pow(1+Math.exp(-z[i]),2);
        }

        return output;
    }
}
