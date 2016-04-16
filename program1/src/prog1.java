/**
 * Sereyvathanak Khorn
 * CSCI 497E
 *
 * Program 1
 * Version 1.0
 * April 14, 2016
 *
 * -train train_x.txt train_y.txt a my.model 2000 2 1
 * train dataset1: -train data/dataset1/train_x.txt data/dataset1/train_y.txt a my.model 2500 500 1
 * train dataset2: -train data/dataset2/train_x.txt data/dataset2/train_y.txt a my.model 2000000 1 1
 * train dataset3: -train data/dataset3/train_x.txt data/dataset3/train_y.txt a my.model 2000 1 8
 */

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.*;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Objects;

public class prog1 {

    public static void main(String[] args) throws IOException {

        if (Objects.equals(args[0], "-train")) {
            if (Objects.equals(args[3], "a")) {
                System.out.println("Training mode a");

                // Training Mode (Analytical Solution)
                trainA(args);

            }
            else if (Objects.equals(args[3], "g")){
                System.out.println("Training mode g");
            }

        }
        else if (Objects.equals(args[0], "-pred")) {
            System.out.println("Prediction mode");
        }
        else if (Objects.equals(args[0], "-eval")) {
            System.out.println("Evaluation mode");
        }
        else {
            System.out.println("Unknown mode");
            System.exit(0);
        }

//        String filex = args[1];
//        String filey = args[2];
//
//        int N = Integer.parseInt(args[5]);
//        int D = Integer.parseInt(args[6]);
//        int K = Integer.parseInt(args[7]);
//
//        RealMatrix xhat = new Array2DRowRealMatrix(new double[N][D+1]);
//        RealMatrix yhat = new Array2DRowRealMatrix(new double[N][1]);
//
//
//        String train_x_reader = fileOpen(filex);
//        String train_y_reader = fileOpen(filey);
//
////        String[] trainXString = train_x_reader.split("\r\n");
////        String[] trainYString = train_y_reader.split("\r\n");
//        String[] trainXString = train_x_reader.split(" \n");
//        String[] trainYString = train_y_reader.split(" \n");
//
//        matrixConvert(trainXString, xhat, D);   // xhat
//        matrixConvert(trainYString, yhat, 0);   // yhat
//
//        // Calculating BetaHat*
//        RealMatrix z = xhat.transpose().multiply(xhat);
//        z = new LUDecomposition(z).getSolver().getInverse();
//        z = z.multiply(xhat.transpose().multiply(yhat));
//
//        System.out.println("Hello world");
//        System.out.println(Arrays.toString(z.getColumn(0)));
    }


    /**
     * Training Mode (Analytical Solution)
     * @param args arguments from terminal
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

//        RealMatrix xhat = new Array2DRowRealMatrix(new double[N][D+1]);
//        RealMatrix yhat = new Array2DRowRealMatrix(new double[N][1]);


        String train_x_reader = fileOpen(filex);
        String train_y_reader = fileOpen(filey);

        // Windows
//        String[] trainXString = train_x_reader.split("\r\n");
//        String[] trainYString = train_y_reader.split("\r\n");
        // Linux
        String[] trainXString = train_x_reader.split(" \n");
        String[] trainYString = train_y_reader.split(" \n");

//        matrixConvert(trainXString, xhat, D);   // xhat
//        matrixConvert(trainYString, yhat, 0);   // yhat

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
        System.out.println(Arrays.toString(z.getColumn(0)));
        writeFile(outFile, z);             // write out the file
    }

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
    public static String fileOpen(String fileName) throws IOException {
        String fileReader = null;
        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            StringBuilder sb = new StringBuilder();
            String line = br.readLine();

            while (line != null) {
                sb.append(line);
                sb.append(System.lineSeparator());
                line = br.readLine();
            }
            fileReader = sb.toString();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return fileReader;
    }

    private static void writeFile(String fileName, RealMatrix matrix) throws IOException {
        Writer writer = null;
        NumberFormat science = new DecimalFormat("0.###E0");
        try {
            writer = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(fileName), "utf-8"));
            writer.write("Something");

            //TODO FIX THIS

            writer.write(science.format(matrix.getEntry(0 ,0)));


        } catch (IOException ex) {
            // report
        } finally {
            assert writer != null;
            writer.close();
            //try {writer.close();} catch (Exception ex) {/*ignore*/}
        }
    }

}