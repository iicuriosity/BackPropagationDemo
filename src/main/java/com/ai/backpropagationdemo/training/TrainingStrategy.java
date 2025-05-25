package com.ai.backpropagationdemo.training;

import com.ai.backpropagationdemo.layer.Layer;

import java.util.ArrayList;

public interface TrainingStrategy {
    void train(ArrayList<Layer> layers);
}
