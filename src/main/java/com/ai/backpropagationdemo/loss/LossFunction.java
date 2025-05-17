package com.ai.backpropagationdemo.loss;

public interface LossFunction {

    public double compute(double[] result, double[] expectedValuer);

    public double[] computeDerivative(double[] result, double[] expectedValuer);
}
