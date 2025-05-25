package com.ai.backpropagationdemo.batch.problems.minmaxrowsum;

import com.ai.backpropagationdemo.batch.Batch;
import lombok.RequiredArgsConstructor;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;

public class MinMaxRowSumMatrixBatch extends Batch {


    private final int batchSize;
    private final int inputLayerSize;
    private final Set<TrainingData> trainingDataSet = new HashSet<>();

    public MinMaxRowSumMatrixBatch(int inputLayerSize, int batchSize) {
        super(inputLayerSize,batchSize);
        this.inputLayerSize = inputLayerSize;
        this.batchSize = batchSize;

        ThreadLocalRandom randomGenerator = ThreadLocalRandom.current();
        for(int i = 0; i < batchSize; i++){
            int  arraySize = Math.abs(randomGenerator.nextInt()%inputLayerSize);
            int[] input = new int[inputLayerSize];
            double[] doubleInput = new double[inputLayerSize];
            for(int j = 0; j < arraySize; j++){
                input[j] =  randomGenerator.nextInt();
                doubleInput[j] = input[j];
            }
            TrainingData data = new TrainingData(doubleInput, new double[]{solve(input)});
            trainingDataSet.add(data);
        }
    }

    @Override
    public Set<TrainingData> getTrainingData() {
        return trainingDataSet;
    }


    public int solve(int[] array) {
        int n = array.length;
        int maxDivisor = (int) Math.sqrt(n);
        int minMaxRowSum = Integer.MAX_VALUE;
        int dimension = 0;

        for (int divisor = 2; divisor <= maxDivisor; divisor++) {
            if (n % divisor == 0) {
                // Check both divisor and n/divisor

                int maxRowSum1 = getMaxRowSum(array, divisor);
                int maxRowSum2 = getMaxRowSum(array, n / divisor);

                if (maxRowSum1 < minMaxRowSum) {
                    minMaxRowSum = maxRowSum1;
                    dimension = divisor;
                }

                if (divisor != n / divisor && maxRowSum2 < minMaxRowSum) {
                    minMaxRowSum = maxRowSum2;
                    dimension = n / divisor;
                }
            }
        }
        return dimension;
    }

    private int getMaxRowSum(int[] array, int numRows) {
        int numCols = array.length / numRows;
        int maxRowSum = Integer.MIN_VALUE;
        int currentRowSum = 0;

        for (int i = 0; i < array.length; i++) {
            currentRowSum += array[i];
            if ((i + 1) % numCols == 0) {
                maxRowSum = Math.max(maxRowSum, currentRowSum);
                currentRowSum = 0;
            }
        }

        return maxRowSum;
    }

}
