package com.ai.backpropagationdemo.utility;

public final class MathUtils {

    private MathUtils() {
        throw new UnsupportedOperationException("Utility class");
    }

    public static double[][] vectorXTransposeMultiply(double[] vector, double[] transpose) {
        if (vector==null|| transpose ==null|| vector.length == 0 || transpose.length == 0) {
            throw new IllegalArgumentException("The vectors must not be null or empty");
        }

        double[][] result = new double[vector.length][transpose.length];
        for (int i = 0; i < vector.length; i++) {
            for (int j = 0; j < transpose.length; j++) {
                result[i][j] = vector[i] * transpose[j];
            }
        }
        return result;
    }

    public static double[] hadamard(double[] firstVector, double[] secondVector) {
        if (firstVector==null|| secondVector ==null|| firstVector.length == 0 || secondVector.length == 0) {
            throw new IllegalArgumentException("The vectors must not be null or empty");
        }
        if ( firstVector.length != secondVector.length) {
            throw new IllegalArgumentException("The vectors must have the same length");
        }

        double[] result = new double[firstVector.length];
        for (int i = 0; i < firstVector.length; i++) {
            result[i] = firstVector[i] * secondVector[i];
        }
        return result;
    }

    public static double[][] transpose(double[][] matrix) {
        if(matrix==null || matrix.length==0 || matrix[0]==null || matrix[0].length==0)
            throw new IllegalArgumentException("The matrix cannot be null or empty");
        double[][] result = new double[matrix[0].length][matrix.length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                result[j][i] = matrix[i][j];
            }
        }

        return result;
    }

    public static double[] matrixXVectorMultiply(double[][] matrix, double[] vector) {
        if(matrix==null || matrix.length==0 || matrix[0]==null || matrix[0].length==0)
            throw new IllegalArgumentException("The matrix cannot be null or empty");

        if(vector==null || vector.length==0 )
            throw new IllegalArgumentException("The vector cannot be null or empty");

        if(matrix[0].length!=vector.length)
            throw new IllegalArgumentException("The matrix's column dimension must be equal to the vector's length");

        double[] result = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }

        return result;
    }
}