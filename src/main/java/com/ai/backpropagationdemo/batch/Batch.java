package com.ai.backpropagationdemo.batch;

import java.util.Arrays;
import java.util.Set;

public abstract class Batch {

    public Batch(int inputLayerSize, int batchSize){}

    public abstract Set<TrainingData> getTrainingData();

    public record TrainingData(double[] inputs, double[] outputs) { }
}
