package com.ai.backpropagationdemo.batch;

import java.util.Set;

public interface Batch {

    Set<TrainingData> getTrainingData();

    public record TrainingData(double[] inputs, double[] outputs) {}
}
