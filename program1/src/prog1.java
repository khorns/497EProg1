/**
 * Sereyvathanak Khorn
 * CSCI 497E
 *
 * Program 1
 * Version 1.0
 * April 14, 2016
 *
 * -train train_x.txt train_y.txt a my.model 2000 2 1
 */

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
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

        int N = Integer.parseInt(args[5]);
        int D = Integer.parseInt(args[6]);
        int K = Integer.parseInt(args[7]);

        RealMatrix xhat = new Array2DRowRealMatrix(new double[N][D+1]);
        RealMatrix yhat = new Array2DRowRealMatrix(new double[N][1]);


        String train_x_reader = fileOpen(filex);
        String train_y_reader = fileOpen(filey);

//        String[] trainXString = train_x_reader.split("\r\n");
//        String[] trainYString = train_y_reader.split("\r\n");
        String[] trainXString = train_x_reader.split(" \n");
        String[] trainYString = train_y_reader.split(" \n");

        matrixConvert(trainXString, xhat, D);   // xhat
        matrixConvert(trainYString, yhat, 0);   // yhat

        // Calculating BetaHat*
        RealMatrix z = xhat.transpose().multiply(xhat);
        z = new LUDecomposition(z).getSolver().getInverse();
        z = z.multiply(xhat.transpose().multiply(yhat));

        System.out.println("Hello world");
        System.out.println(Arrays.toString(z.getColumn(0)));
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

}