package com.ai.backpropagationdemo.activation;

public interface ActivationFunction {

    public double[] apply(double[] x);

    public double[] derivative(double[] x);
}
