package com.ai.backpropagationdemo;

import com.ai.backpropagationdemo.activation.SigmoidActivation;
import com.ai.backpropagationdemo.batch.problems.minmaxrowsum.MinMaxRowSumMatrixBatch;
import com.ai.backpropagationdemo.loss.MeanSquaredErrorLoss;
import com.ai.backpropagationdemo.network.UniformActivationNetwork;
import com.ai.backpropagationdemo.training.DefaultTrainingStrategy;
import com.ai.backpropagationdemo.training.TrainingStrategy;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.util.Scanner;

@SpringBootApplication
public class BackPropagationDemoApplication {

    public static void main(String[] args) {
        SpringApplication.run(BackPropagationDemoApplication.class, args);
        Scanner scanner = new Scanner(System.in);
        System.out.println("insert array size:");
        int arraySize = scanner.nextInt();
        System.out.println("insert batch size:");
        int batchSize = scanner.nextInt();
        System.out.println("insert number of iterations:");
        int numberOfIterations = scanner.nextInt();
        System.out.println("insert number of batches per iteration:");
        int numberOfBatchesPerIteration = scanner.nextInt();
        System.out.println("insert number of layers:");
        int numberOfLayers = scanner.nextInt();
        System.out.println("insert number of perceptrons in each layer:");
        int[] perceptronsAtEachLayer = new int[numberOfLayers];
        for(int i = 0; i < numberOfLayers; i++){
            perceptronsAtEachLayer[i] = scanner.nextInt();
        }
        TrainingStrategy strategy = new DefaultTrainingStrategy(0.1, new MeanSquaredErrorLoss(),batchSize,numberOfBatchesPerIteration,numberOfIterations, MinMaxRowSumMatrixBatch.class);
        UniformActivationNetwork network = new UniformActivationNetwork(arraySize, numberOfLayers, perceptronsAtEachLayer, new SigmoidActivation(), strategy);
        System.out.println("Training network...");
        network.train();
        System.out.println("Network trained successfully!");
        System.out.println("Testing network...");
        int[] input = new int[arraySize];
        double[] inputDouble = new double[arraySize];
        for(int i = 0; i < arraySize; i++){
            System.out.println("insert input["+i+"]:");
            input[i] = scanner.nextInt();
            inputDouble[i] = input[i];
        }
        System.out.println("Calculating...");
        double[] result = network.forward(inputDouble);
        for(int i = 0; i < result.length; i++){
            System.out.println("Result["+i+"] = "+result[i]);
        }
        MinMaxRowSumMatrixBatch batch = new MinMaxRowSumMatrixBatch(arraySize,batchSize);
        System.out.println("Expected result: "+ batch.solve(input));

    }

}
