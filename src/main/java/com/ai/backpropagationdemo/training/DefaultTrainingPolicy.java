package com.ai.backpropagationdemo.training;

import com.ai.backpropagationdemo.layer.Layer;
import com.ai.backpropagationdemo.loss.LossFunction;
import lombok.Getter;

import java.util.ArrayList;

public class DefaultTrainingPolicy implements TrainingPolicy{

    private double learningRate;
    private LossFunction lossFunction;

    public DefaultTrainingPolicy(double learningRate, LossFunction lossFunction) {
        this.learningRate = learningRate;
        this.lossFunction = lossFunction;
    }

    @Override
    public void train(ArrayList<Layer> layers) {

    }
}
