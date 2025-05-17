package com.ai.backpropagationdemo.training;

import com.ai.backpropagationdemo.layer.Layer;

import java.util.ArrayList;

public interface TrainingPolicy {
    void train(ArrayList<Layer> layers);
}
