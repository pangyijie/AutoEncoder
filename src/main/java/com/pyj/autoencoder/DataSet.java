package com.pyj.autoencoder;

import java.io.*;

public class DataSet {
    public double[][] train_X = new double[30][13];
    public int[][] train_Y = new int[30][3];
    public double[][] test_X = new double[10][13];

    //获取训练数据
    public void getTrainData() throws IOException {
        BufferedReader br = new BufferedReader(
                new InputStreamReader(
                        new FileInputStream("D:\\train_set.txt")));
        for (int i = 0; i < 30; i++) {
            String[] str = br.readLine().split(",");
            double tag = Double.parseDouble(str[13]);
            for (int j = 0; j < 13; j++) {
                this.train_X[i][j] = Double.parseDouble(str[j]);
            }
            if (tag == 0.1) {
                this.train_Y[i][0] = 1;
                this.train_Y[i][1] = 0;
                this.train_Y[i][2] = 0;
            } else if (tag == 0.5) {
                this.train_Y[i][0] = 0;
                this.train_Y[i][1] = 1;
                this.train_Y[i][2] = 0;
            } else {
                this.train_Y[i][0] = 0;
                this.train_Y[i][1] = 0;
                this.train_Y[i][2] = 1;
            }
        }
    }

    //获取测试数据
    public void getTestData() throws IOException {
        BufferedReader br = new BufferedReader(
                new InputStreamReader(
                        new FileInputStream("D:\\test_set.txt")));
        for (int i = 0; i < 10; i++) {
            String[] str = br.readLine().split(",");
            for (int j = 0; j < 13; j++) {
                this.test_X[i][j] = Double.parseDouble(str[j]);
            }
        }
    }
}
