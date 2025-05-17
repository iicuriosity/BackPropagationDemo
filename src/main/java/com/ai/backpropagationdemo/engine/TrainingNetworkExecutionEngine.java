package com.ai.backpropagationdemo.engine;

import com.ai.backpropagationdemo.batch.Batch;
import com.ai.backpropagationdemo.layer.Layer;
import com.ai.backpropagationdemo.loss.LossFunction;
import com.ai.backpropagationdemo.utility.MathUtils;
import lombok.RequiredArgsConstructor;
import org.apache.commons.lang3.tuple.Pair;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

@RequiredArgsConstructor
public class TrainingNetworkExecutionEngine {


    private final ArrayList<Layer> layers;
    private final Batch batch;
    private final LossFunction lossFunction;
    private final Map<TrainingDataLayer, TrainingLayerState> DataLayerStates = new HashMap<>();

    private double loss;
    private boolean trainingProcessed;

    public double getLoss() {
        if (!trainingProcessed)
            throw new IllegalStateException("Training process not yet started");
        return loss;
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
        // TODO adjust weights
    }

    private void forwardRun() {
        for (Batch.TrainingData trainingData : batch.getTrainingData()) {
            double[] layerInput = trainingData.inputs();
            for (Layer layer : layers) {
                Pair<double[], double[]> weightedInputAndOutput = layer.forward(layerInput);
                layerInput = weightedInputAndOutput.getRight();
                TrainingLayerState state = new TrainingLayerState(weightedInputAndOutput.getLeft(), layerInput, null, null, null);
                DataLayerStates.put(new TrainingDataLayer(trainingData, layer), state);
            }
        }
    }

    private void computeLoss() {
        double cumulative = 0;
        for (Batch.TrainingData td : batch.getTrainingData()) {
            Layer lastLayer = layers.getLast();
            TrainingLayerState st = DataLayerStates.get(new TrainingDataLayer(td, lastLayer));
            cumulative += lossFunction.compute(st.h(), td.outputs());
        }
        this.loss = cumulative / batch.getTrainingData().size();
    }

    private void backPropagate() {

        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer currentLayer = layers.get(i);
            for (Batch.TrainingData trainingData : batch.getTrainingData()) {
                TrainingLayerState currentState = DataLayerStates.get(new TrainingDataLayer(trainingData, currentLayer));

                double[] computeVector = null;
                if (currentLayer==layers.getLast())
                    computeVector = lossFunction.computeDerivative(currentState.h(), trainingData.outputs());
                else{
                    TrainingLayerState nextState = DataLayerStates.get(new TrainingDataLayer(trainingData, layers.get(i + 1)));
                    computeVector = MathUtils.matrixXVectorMultiply(MathUtils.transpose(layers.get(i + 1).getWeights()),nextState.signal());
                }

                double[] signal = MathUtils.hadamard(computeVector, currentLayer.derivative(currentState.z()));

                double[] previousOutput = layers.size() > 1 ? trainingData.inputs() :
                        DataLayerStates.get(new TrainingDataLayer(trainingData, layers.get(layers.size() - 2))).h();

                double[][] weightGradient = MathUtils.vectorXTransposeMultiply(signal, previousOutput);

                TrainingLayerState newState = new TrainingLayerState(currentState.z(), currentState.h(), signal, weightGradient, signal);

                DataLayerStates.put(new TrainingDataLayer(trainingData, currentLayer), newState);
            }
        }

    }


    private record TrainingDataLayer(Batch.TrainingData trainingData, Layer layer) {
    }

    private record TrainingLayerState(double[] z, double[] h, double[] signal, double[][] dW, double[] db) {
    }


}
