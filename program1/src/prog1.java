/**
 * Sereyvathanak Khorn
 * CSCI 497E
 *
 * Program 1
 * Version 1.0
 * April 14, 2016
 *
 *  Training Analytical Solution
 * -train train_x.txt train_y.txt a my.model 2000 2 1
 * train dataset1: -train data/dataset1/train_x.txt data/dataset1/train_y.txt a my.model 2500 500 1
 * train dataset2: -train data/dataset2/train_x.txt data/dataset2/train_y.txt a my.model 2000000 1 1
 * train dataset3: -train data/dataset3/train_x.txt data/dataset3/train_y.txt a my.model 2000 1 8
 *
 *  Prediction
 * -pred train_x.txt my.model my.prediction 2000 2 1
 * predict dataset1: -pred data/dataset1/train_x.txt my.model my.prediction 2500 500 1
 * predict dataset2: -pred data/dataset2/train_x.txt my.model my.prediction 2000000 1 1
 * predict dataset3: -pred data/dataset3/train_x.txt my.model my.prediction 2000 1 8
 *
 *  Evaluation
 * evaluation dataset4 -eval data/dataset4/dev_x.txt data/dataset4/dev_y.txt my.model 500 2 1
 * evaluation dataset1 -eval data/dataset1/dev_x.txt data/dataset1/dev_y.txt my.model 500 500 1
 * evaluation dataset3 -eval data/dataset3/dev_x.txt data/dataset3/dev_y.txt my.model 500 1 8
 *
 *  Training Gradient Descent Solution
 *  train dataset4: -train data/dataset4/train_x.txt data/dataset4/train_y.txt g 0.1 0.00001 descent.model 2000 2 1
 *  train dataset1: -train data/dataset1/train_x.txt data/dataset1/train_y.txt g 0.1 0.05 descent.model 2500 500 1
 *  train dataset2: -train data/dataset2/train_x.txt data/dataset2/train_y.txt g 0.001 0.05 descent.model 2000000 1 1
 *  train dataset3: -train data/dataset3/train_x.txt data/dataset3/train_y.txt g 0.0000005 0.0001 descent.model 2000 1 8
 */

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.*;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Objects;

public class prog1 {

    public static void main(String[] args) throws IOException {

        if (Objects.equals(args[0], "-train")) {
            if (Objects.equals(args[3], "a")) {         // Training Mode (Analytical Solution)
                System.out.println("Training Mode (Analytical Solution)");

                double startTime = System.nanoTime();
                trainA(args);
                double endTime = System.nanoTime();
                System.out.println((endTime - startTime) / 1000000000.0);
            }
            else if (Objects.equals(args[3], "g")){     // Training mode (Gradient Descent)
                System.out.println("Training mode (Gradient Descent)");

                double startTime = System.nanoTime();
                trainG(args);
                double endTime = System.nanoTime();
                System.out.println((endTime - startTime) / 1000000000.0);
            }
        }
        else if (Objects.equals(args[0], "-pred")) {    // Predictions Mode
            System.out.println("Prediction mode");

            prediction(args);
        }
        else if (Objects.equals(args[0], "-eval")) {    // Evaluation Mode
            System.out.println("Evaluation mode");
            evaluation(args);
        }
        else {
            System.out.println("Unknown mode");
            System.exit(0);
        }
    }


    /**
     * Train Mode using Gradient Descent Solution
     * @param args Command line arguments
     * @throws IOException
     */
    private static void trainG(String[] args) throws IOException {

        String filex = "", filey = "", outFile = "";
        double ss = 0, st = 0;
        int N = 0, D = 0, K = 0;

        //
        try {
            filex = args[1];
            filey = args[2];
            outFile = args[6];

            ss = Double.parseDouble(args[4]);
            st = Double.parseDouble(args[5]);

            N = Integer.parseInt(args[7]);
            D = Integer.parseInt(args[8]);
            K = Integer.parseInt(args[9]);
        }
        catch (Exception e) {
            System.out.println("Fail Conversion or wrong number of parameters, check your parameters");
            System.exit(1);
        }

        RealMatrix xhat;
        RealMatrix yhat;
        RealMatrix bhat;

        System.out.println("Running...");

        String[] trainXString = fileOpen(filex);
        String[] trainYString = fileOpen(filey);

        // Create matrix for xhat and yhat
        if (K == 1) {
            xhat = new Array2DRowRealMatrix(new double[N][D+1]);
            yhat = new Array2DRowRealMatrix(new double[N][1]);
            bhat = new Array2DRowRealMatrix(new double[D+1][1]);   // beta hat
            matrixConvert(trainXString, xhat, D);   // xhat
            matrixConvert(trainYString, yhat, 0);   // yhat
        }
        else {
            xhat = new Array2DRowRealMatrix(new double[N][K+1]);
            yhat = new Array2DRowRealMatrix(new double[N][1]);
            bhat = new Array2DRowRealMatrix(new double[K+1][1]);   // beta hat
            matrixConvertK(trainXString, xhat, K);   // xhat
            matrixConvert(trainYString, yhat, 0);   // yhat
        }

        boolean converged = false;
        RealMatrix risk;
        RealMatrix newRisk;
        RealMatrix gradient = xhat.transpose().multiply(xhat.multiply(bhat).subtract(yhat)).scalarMultiply(2.0/N);

        risk = yhat.subtract(xhat.multiply(bhat)).transpose().multiply(yhat.subtract(xhat.multiply(bhat)));
        double ERisk = risk.getEntry(0,0) / N;
        double newERisk;

        while (!converged) {
            bhat = bhat.subtract(gradient.scalarMultiply(ss));
            newRisk = yhat.subtract(xhat.multiply(bhat)).transpose().multiply(yhat.subtract(xhat.multiply(bhat)));
            newERisk = newRisk.getEntry(0,0) / N;

            converged = checkConvergence(ERisk, newERisk, st);      // Checking for convergence
            ERisk = newERisk;
        }

        writeFile(outFile, bhat);
        System.out.println("Complete");

    }


    /**
     * Checking convergence
     * @param ERisk Old objective value
     * @param newERisk new objective value
     * @param st threshold
     * @return TRUE if convergence
     */
    private static boolean checkConvergence(double ERisk, double newERisk, double st) {
        double relativeReduction = (ERisk - newERisk) / ERisk;
        return relativeReduction <= st;
    }


    /**
     * Evaluation Mode
     * @param args Command line arguments
     * @throws IOException
     */
    private static void evaluation(String[] args) throws IOException {

        String filex, filey, fileModel;
        filex = filey = fileModel = "";
        int N = 0, D = 0, K = 0;

        try {
            filex = args[1];
            filey = args[2];
            fileModel = args[3];

            N = Integer.parseInt(args[4]);
            D = Integer.parseInt(args[5]);
            K = Integer.parseInt(args[6]);
        }
        catch (Exception e) {
            System.out.println("Fail Conversion or wrong number of parameters, check your parameters");
            System.exit(1);
        }

        String[] filexReader = fileOpen(filex);
        String[] modelReaderA = fileOpen(fileModel);
        String[] fileyReader = fileOpen(filey);

        double[][] filexArray;
        double[] fileModelArray;
        double[] fileyArray = new double[N];

        String[] modelReader = modelReaderA[0].split("\n");     //For Linux (Check on this if running on windows)
        System.out.println("Running...");
        if (K == 1) {
            filexArray = new double[N][D];
            fileModelArray = new double[D+1];
        }
        else {
            filexArray = new double[N][K+1];
            fileModelArray = new double[K+1];
        }

        // filey
        for (int i = 0; i < N; i++) {
            fileyArray[i] = Double.parseDouble(fileyReader[i]);
        }

        // filex
        String[] tempArray;
        if (D >= 2) {
            for (int i = 0; i < N; i++) {
                tempArray = filexReader[i].split("\\s");

                for (int j = 0; j < D; j++) {
                    filexArray[i][j] = Double.parseDouble(tempArray[j]);
                }
            }
        }
        else if(D == 1 && K > 1) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j <= K; j++) {
                    filexArray[i][j] = Math.pow(Double.parseDouble(filexReader[i]), j);
                }
            }
        }
        else {
            for (int i = 0; i < N; i++) {
                filexArray[i][0] = Double.parseDouble(filexReader[i]);
            }
        }

        // fileModel
        if (K == 1) {
            for (int i = 0; i < D + 1; i++) {
                fileModelArray[i] = Double.parseDouble(modelReader[i]);
            }
        }
        else {
            for (int i = 0; i < K + 1; i++) {
                fileModelArray[i] = Double.parseDouble(modelReader[i]);
            }
        }

        // Calculating Mean Square Error
        double result = 0;
        double pred[] = new double[N];

        if (K == 1) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < D; j++) {
                    result += filexArray[i][j] * fileModelArray[j+1];
                }
                result += fileModelArray[0];
                pred[i] = result;
                result = 0;
            }
        }
        else {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j <= K; j++) {
                    result += filexArray[i][j] * fileModelArray[j];
                }
                pred[i] = result;
                result = 0;
            }
        }

        NumberFormat science = new DecimalFormat("0.###E0");

        double total = 0;
        for (int i = 0; i < N; i++) {
            System.out.println(science.format(pred[i]) + " : " + science.format(fileyArray[i]));
            total += Math.pow(pred[i] - fileyArray[i], 2);
        }
        total = total / N;
        System.out.println("MSE is: " + total);
    }

    /**
     * Prediction Mode
     * @param args command line arguments
     * @throws IOException
     */
    private static void prediction(String[] args) throws IOException {

        String filex, fileModel, filePred;
        filex =  fileModel = filePred = "";
        int N = 0, D = 0, K = 0;

        try {
            filex = args[1];
            fileModel = args[2];
            filePred = args[3];

            N = Integer.parseInt(args[4]);
            D = Integer.parseInt(args[5]);
            K = Integer.parseInt(args[6]);
        }
        catch (Exception e) {
            System.out.println("Fail Conversion or wrong number of parameters, check your parameters");
            System.exit(1);
        }

        String[] filexReader = fileOpen(filex);
        String[] modelReaderA = fileOpen(fileModel);

        double[][] filexArray;
        double[] fileModelArray;

        String[] modelReader = modelReaderA[0].split("\n");     //Linux compatibility
        System.out.println("Running...");
        if (K == 1) {
            filexArray = new double[N][D];
            fileModelArray = new double[D+1];
        }
        else {
            filexArray = new double[N][K+1];
            fileModelArray = new double[K+1];
        }

        // filex
        String[] tempArray;
        if (D >= 2) {
            for (int i = 0; i < N; i++) {
                tempArray = filexReader[i].split("\\s");

                for (int j = 0; j < D; j++) {
                    filexArray[i][j] = Double.parseDouble(tempArray[j]);
                }
            }
        }
        else if(D == 1 && K > 1) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j <= K; j++) {
                    filexArray[i][j] = Math.pow(Double.parseDouble(filexReader[i]), j);
                }
            }
        }
        else {
            for (int i = 0; i < N; i++) {
                filexArray[i][0] = Double.parseDouble(filexReader[i]);
            }
        }

        // fileModel
        if (K == 1) {
            for (int i = 0; i < D + 1; i++) {
                fileModelArray[i] = Double.parseDouble(modelReader[i]);
            }
        }
        else {
            for (int i = 0; i < K + 1; i++) {
                fileModelArray[i] = Double.parseDouble(modelReader[i]);
            }
        }

        double result = 0;
        double pred[] = new double[N];

        if (K == 1) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < D; j++) {
                    result += filexArray[i][j] * fileModelArray[j+1];
                }
                result += fileModelArray[0];
                pred[i] = result;
                result = 0;
            }
        }
        else {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j <= K; j++) {
                    result += filexArray[i][j] * fileModelArray[j];
                }
                pred[i] = result;
                result = 0;
            }
        }

        writePredictFile(filePred, pred);
        System.out.println("Complete");
    }

    /**
     * Training Mode (Analytical Solution)
     * @param args arguments from terminal as input
     * @throws IOException
     */
    private static void trainA (String[] args) throws IOException {
        String filex = args[1];
        String filey = args[2];
        String outFile = args[4];

        int N = Integer.parseInt(args[5]);
        int D = Integer.parseInt(args[6]);
        int K = Integer.parseInt(args[7]);
        RealMatrix xhat;
        RealMatrix yhat;

        System.out.println("Running...");

        String[] trainXString = fileOpen(filex);
        String[] trainYString = fileOpen(filey);

        // Create matrix for xhat and yhat
        if (K == 1) {
            xhat = new Array2DRowRealMatrix(new double[N][D+1]);
            yhat = new Array2DRowRealMatrix(new double[N][1]);
            matrixConvert(trainXString, xhat, D);   // xhat
            matrixConvert(trainYString, yhat, 0);   // yhat
        }
        else {
            xhat = new Array2DRowRealMatrix(new double[N][K+1]);
            yhat = new Array2DRowRealMatrix(new double[N][1]);
            matrixConvertK(trainXString, xhat, K);   // xhat
            matrixConvert(trainYString, yhat, 0);   // yhat
        }

        // Calculating BetaHat*
        RealMatrix z = xhat.transpose().multiply(xhat);
        z = new LUDecomposition(z).getSolver().getInverse();
        z = z.multiply(xhat.transpose().multiply(yhat));

        System.out.println("Completed");
        writeFile(outFile, z);             // write out the file
    }

    /**
     * Convert an array to a RealMatrix
     * @param data data for matrix
     * @param MatData matrix result
     * @param K poly order
     */
    private static void matrixConvertK(String[] data, RealMatrix MatData, int K) {
        double doubleTemp;
        double original;
        int index = 0;
        int DTemp = 0;

        for (String aTrainString : data) {
            doubleTemp = Double.parseDouble(aTrainString);
            original = doubleTemp;
            while (DTemp <= K) {    // Create a row matrix, eg: [1, xxx, xxx]
                if (DTemp == 0) {
                    MatData.setEntry(index, 0, 1);
                }
                else {
                    doubleTemp = Math.pow(original, DTemp);
                    MatData.setEntry(index, DTemp, doubleTemp);
                }
                DTemp++;
            }
            index++;
            DTemp = 0;
        }
    }


    /**
     * Build matrix
     * @param data string that hold the data for building matrix
     * @param MatData matrix to be build
     * @param D dimension
     */
    private static void matrixConvert(String[] data, RealMatrix MatData, int D) {
        double doubleTemp;
        int index = 0;
        int DTemp = 0;
        String[] rowDivider;
        for (String aTrainString : data) {

            if (D >= 2) {
                rowDivider = aTrainString.split("\\s");

                while (DTemp <= D) {    // Create a row matrix, eg: [1, xxx, xxx]
                    if (DTemp == 0) {
                        MatData.setEntry(index, 0, 1);
                    }
                    else {
                        MatData.setEntry(index, DTemp, Double.parseDouble(rowDivider[DTemp-1]));
                    }
                    DTemp++;
                }
                index++;
                DTemp = 0;
            }
            else {
                doubleTemp = Double.parseDouble(aTrainString);

                if (D == 0) {
                    MatData.setEntry(index, D, doubleTemp);
                    index++;
                }
                else {
                    while (DTemp <= D) {    // Create a row matrix, eg: [1, xxx, xxx]
                        if (DTemp == 0) {
                            MatData.setEntry(index, 0, 1);
                        }
                        else {
                            MatData.setEntry(index, DTemp, doubleTemp);
                        }
                        DTemp++;
                    }
                    index++;
                    DTemp = 0;
                }
            }
        }
    }


    /**
     * Open up a file and read everything into a string
     * @param fileName the file name
     * @return a string
     * @throws IOException
     */
    public static String[] fileOpen(String fileName) throws IOException {
        String fileReader = null;
        String[] spliter = new String[0];
        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            StringBuilder sb = new StringBuilder();
            String line = br.readLine();

            while (line != null) {
                sb.append(line);
                sb.append(System.lineSeparator());
                line = br.readLine();
            }
            fileReader = sb.toString();

            // Windows
//            spliter = fileReader.split("\r\n");
            // Linux
            spliter = fileReader.split(" \n");

        } catch (IOException e) {
            e.printStackTrace();
        }
        return spliter;
    }


    /**
     * Write a file for training mode
     * @param fileName file name
     * @param matrix matrix input
     * @throws IOException
     */
    private static void writeFile(String fileName, RealMatrix matrix) throws IOException {
        Writer writer = null;
        NumberFormat science = new DecimalFormat("0.###E0");
        try {
            writer = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(fileName), "utf-8"));

            // Write to a file base on column
            for (int i = 0; i < matrix.getData().length; i++) {
                writer.write(science.format(matrix.getEntry(i ,0)) + "\n");
            }
        } catch (IOException ignored) {
        } finally {
            assert writer != null;
            writer.close();
        }
    }


    /**
     * Writing a file for prediction
     * @param fileName file name
     * @param array the prediction data
     * @throws IOException
     */
    private static void writePredictFile(String fileName, double[] array) throws IOException {
        Writer writer = null;
        NumberFormat science = new DecimalFormat("0.###E0");
        try {
            writer = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(fileName), "utf-8"));

            // Write to a file base on column
            for (int i = 0; i < array.length; i++) {
                writer.write(science.format(array[i]) + "\n");
            }
        } catch (IOException ignored) {

        } finally {
            assert writer != null;
            writer.close();
        }
    }
}