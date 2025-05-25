package com.ai.backpropagationdemo.training;

import com.ai.backpropagationdemo.batch.Batch;
import com.ai.backpropagationdemo.layer.Layer;
import com.ai.backpropagationdemo.loss.LossFunction;
import com.ai.backpropagationdemo.utility.MathUtils;
import lombok.RequiredArgsConstructor;
import org.apache.commons.lang3.tuple.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;


@RequiredArgsConstructor
public class TrainingRun {


    private final ArrayList<Layer> layers;
    private final Batch batch;
    private final LossFunction lossFunction;
    private final double learningRate;

    private double loss;
    private boolean trainingProcessed;

    private final Map<TrainingDataLayer, TrainingLayerState> dataLayerStates = new ConcurrentHashMap<>();
    private final Map<Layer, double[][]> layerWeightAdjustmentMap = new ConcurrentHashMap<>();
    private final Map<Layer, double[]> layerBiasAdjustmentMap = new ConcurrentHashMap<>();


    public double getLoss() {
        if (!trainingProcessed)
            throw new IllegalStateException("Training process not yet started");
        return loss;
    }

    public double[][] getLayerWeightAdjustment(Layer layer){
        if (!trainingProcessed)
            throw new IllegalStateException("Training process not yet started");
        if(!layerWeightAdjustmentMap.containsKey(layer))
            throw new IllegalArgumentException("Layer not found in training run");
        return layerWeightAdjustmentMap.get(layer);
    }

    public double[] getLayerBiasAdjustment(Layer layer){
        if (!trainingProcessed)
            throw new IllegalStateException("Training process not yet started");
        if(!layerBiasAdjustmentMap.containsKey(layer))
            throw new IllegalArgumentException("Layer not found in training run");
        return layerBiasAdjustmentMap.get(layer);
    }

    public void execute() {
        if (trainingProcessed)
            throw new IllegalStateException("Training process has already been executed");
        trainingProcessed = true;
        // forward run
        forwardRun();

        // loss calculation
        computeLoss();
        // back propagate
        backPropagate();

        // adjust weights
        adjustWeights();

    }

    private void adjustWeights() {
        for (Layer layer : layers) {
            double[][] batchWeightGradient = new double[layer.getWeights().length][layer.getWeights()[0].length];
            double[] batchBiasGradient = new double[layer.getBiases().length];
            for (Batch.TrainingData trainingData : batch.getTrainingData()) {
                TrainingLayerState state = dataLayerStates.get(new TrainingDataLayer(trainingData, layer));
                for (int i = 0; i < batchWeightGradient.length; i++) {
                    for (int j = 0; j < batchWeightGradient[i].length; j++) {
                        batchWeightGradient[i][j] -= learningRate*state.dW()[i][j];
                    }
                    batchBiasGradient[i] -= learningRate*state.db()[i];
                }
            }

            for (int i = 0; i < batchWeightGradient.length; i++) {
                for (int j = 0; j < batchWeightGradient[i].length; j++) {
                    batchWeightGradient[i][j] /= batch.getTrainingData().size();
                }
                batchBiasGradient[i] /= batch.getTrainingData().size();
            }
            layerWeightAdjustmentMap.put(layer,batchWeightGradient);
            layerBiasAdjustmentMap.put(layer,batchBiasGradient);
        }
    }

    private void forwardRun() {
        for (Batch.TrainingData trainingData : batch.getTrainingData()) {
            double[] layerInput = trainingData.inputs();
            for (Layer layer : layers) {
                Pair<double[], double[]> weightedInputAndOutput = layer.forward(layerInput);
                layerInput = weightedInputAndOutput.getRight();
                TrainingLayerState state = new TrainingLayerState(weightedInputAndOutput.getLeft(), layerInput, null, null, null);
                dataLayerStates.put(new TrainingDataLayer(trainingData, layer), state);
            }
        }
    }

    private void computeLoss() {
        double cumulative = 0;
        for (Batch.TrainingData td : batch.getTrainingData()) {
            Layer lastLayer = layers.getLast();
            TrainingLayerState st = dataLayerStates.get(new TrainingDataLayer(td, lastLayer));
            cumulative += lossFunction.compute(st.h(), td.outputs());
        }
        this.loss = cumulative / batch.getTrainingData().size();
    }

    private void backPropagate() {

        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer currentLayer = layers.get(i);
            for (Batch.TrainingData trainingData : batch.getTrainingData()) {
                TrainingLayerState currentState = dataLayerStates.get(new TrainingDataLayer(trainingData, currentLayer));

                double[] costFunctionGradient = null;
                if (currentLayer==layers.getLast())
                    costFunctionGradient = lossFunction.computeDerivative(currentState.h(), trainingData.outputs());
                else{
                    TrainingLayerState nextState = dataLayerStates.get(new TrainingDataLayer(trainingData, layers.get(i + 1)));
                    costFunctionGradient = MathUtils.matrixXVectorMultiply(MathUtils.transpose(layers.get(i + 1).getWeights()),nextState.signal());
                }

                double[] signal = MathUtils.hadamard(costFunctionGradient, currentLayer.derivative(currentState.z()));

                double[] previousOutput = i ==0 ? trainingData.inputs() :
                        dataLayerStates.get(new TrainingDataLayer(trainingData, layers.get(i - 1))).h();

                double[][] weightGradient = MathUtils.vectorXTransposeMultiply(signal, previousOutput);

                TrainingLayerState newState = new TrainingLayerState(currentState.z(), currentState.h(), signal, weightGradient, signal);

                dataLayerStates.put(new TrainingDataLayer(trainingData, currentLayer), newState);
            }
        }

    }


    private record TrainingDataLayer(Batch.TrainingData trainingData, Layer layer) {
        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (!(obj instanceof TrainingDataLayer other)) return false;

            // layers must be identical
            if (!layer.equals(other.layer)) return false;

            return trainingData.equals(other.trainingData);


            // compare the TrainingData *by content*
        }

        @Override
        public int hashCode() {
            int result = layer.hashCode();
            result = 31 * result + trainingData.hashCode();
            return result;
        }

    }

    private record TrainingLayerState(double[] z, double[] h, double[] signal, double[][] dW, double[] db) {
    }


}
