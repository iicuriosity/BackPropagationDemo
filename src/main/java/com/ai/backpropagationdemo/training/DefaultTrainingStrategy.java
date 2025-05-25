package com.ai.backpropagationdemo.training;

import com.ai.backpropagationdemo.batch.Batch;
import com.ai.backpropagationdemo.layer.Layer;
import com.ai.backpropagationdemo.loss.LossFunction;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class DefaultTrainingStrategy implements TrainingStrategy {

    private double learningRate;
    private final LossFunction lossFunction;
    private final int batchSize;
    private final int batchesPerIteration;
    private final int numberOfIterations;
    private final Constructor<? extends Batch> batchConstructor;

    private double previousAvgLoss = Double.POSITIVE_INFINITY;

    private static final double LR_INCREASE_FACTOR = 1.05;

    private static final double LR_DECREASE_FACTOR = 0.7;

    private static final double LR_MIN = 1e-5;
    private static final double LR_MAX = 1.0;



    public DefaultTrainingStrategy(double initialLearningRate, LossFunction lossFunction, int batchSize,
                                   int batchesPerIteration, int numberOfIterations, Class<? extends Batch> batchClass) {
        this.learningRate = initialLearningRate;
        this.lossFunction = lossFunction;
        this.batchSize = batchSize;
        this.batchesPerIteration = batchesPerIteration;
        this.numberOfIterations = numberOfIterations;
        try {
            this.batchConstructor               = batchClass.getConstructor(int.class, int.class);
        } catch (NoSuchMethodException e) {
            throw new RuntimeException(e);
        }

    }

    @Override
    public void train(ArrayList<Layer> layers) {
        for(int i=0; i<numberOfIterations; i++){
            runIteration(layers);
        }
    }

    public void runIteration(ArrayList<Layer> layers) {
        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {

            List<Future<TrainingRun>> futures = new ArrayList<>(batchesPerIteration);
            for (int i = 0; i < batchesPerIteration; i++) {
                Batch batch = batchConstructor.newInstance(
                        layers.getFirst().getInputDimension(), batchSize);
                TrainingRun run = new TrainingRun(layers, batch, lossFunction, learningRate);
                futures.add(executor.submit(() -> { run.execute(); return run; }));
            }

            List<TrainingRun> finished = new ArrayList<>(futures.size());
            for (Future<TrainingRun> f : futures) {
                finished.add(f.get());
            }

            for (Layer layer : layers) {
                double[][] dw = new double[layer.getWeights().length][layer.getWeights()[0].length];
                double[]   db = new double[layer.getBiases().length];

                for (TrainingRun r : finished) {
                    double[][] gdw = r.getLayerWeightAdjustment(layer);
                    double[]   gdb = r.getLayerBiasAdjustment(layer);
                    for (int rIdx = 0; rIdx < dw.length; rIdx++) {
                        for (int cIdx = 0; cIdx < dw[rIdx].length; cIdx++) {
                            dw[rIdx][cIdx] += gdw[rIdx][cIdx];
                        }
                    }
                    for (int i = 0; i < db.length; i++) db[i] += gdb[i];
                }
                int denom = finished.size();
                for (int rIdx = 0; rIdx < dw.length; rIdx++) {
                    for (int cIdx = 0; cIdx < dw[rIdx].length; cIdx++) {
                        dw[rIdx][cIdx] /= denom;
                    }
                }
                for (int i = 0; i < db.length; i++) db[i] /= denom;

                layer.adjust(dw, db);
            }

            double avgLoss = finished.stream()
                    .mapToDouble(TrainingRun::getLoss)
                    .average()
                    .orElseThrow();

            if (avgLoss < previousAvgLoss) {
                learningRate = Math.min(learningRate * LR_INCREASE_FACTOR, LR_MAX);
            } else {
                learningRate = Math.max(learningRate * LR_DECREASE_FACTOR, LR_MIN);
            }
            previousAvgLoss = avgLoss;
        }
        catch (InstantiationException | InvocationTargetException |
               IllegalAccessException | InterruptedException | ExecutionException ex) {
            throw new RuntimeException(ex);
        }

    }


}
