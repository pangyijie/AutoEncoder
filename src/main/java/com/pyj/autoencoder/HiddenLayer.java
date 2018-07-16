package com.pyj.autoencoder;

import java.util.Random;

//隐藏层
public class HiddenLayer {
    public int N;//样本容量
    public int n_in;//输入维度
    public int n_out;//输出维度
    public double[][] W;//权值矩阵
    public double[] b;//偏置向量
    public Random rng;//随机变量

    //返回max和min之间的随机数
    public double uniform(double min, double max) {
        return rng.nextDouble() * (max - min) + min;
    }

    //以二项分布随机擦除数据，随机置0或1
/*    public int binomial(int n, double p) {
        if (p < 0 || p > 1) {
            return 0;
        }
        int c = 0;
        double r;
        for (int i = 0; i < n; i++) {
            //0.0~1.0随机数
            r = rng.nextDouble();
            if (r < p) c++;
        }
        return c;
    }*/

    //sigmoid函数
    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.pow(Math.E, -x));
    }

    //构造函数
    public HiddenLayer(int N, int n_in, int n_out, double[][] W, double[] b, Random rng) {
        this.N = N;
        this.n_in = n_in;
        this.n_out = n_out;

        if (rng == null) {
            this.rng = new Random(1234);
        } else {
            this.rng = rng;
        }

        if (W == null) {
            this.W = new double[n_out][n_in];
            double a = 1.0 / this.n_in;

            for (int i = 0; i < n_out; i++) {
                for (int j = 0; j < n_in; j++) {
                    this.W[i][j] = uniform(-a, a);
                }
            }
        } else {
            this.W = W;
        }

        if (b == null) {
            this.b = new double[n_out];
        } else {
            this.b = b;
        }
    }

    //线性输出
    public double output(double[] input, double[] w, double b) {
        double linear_output = 0.0;
        for (int j = 0; j < n_in; j++) {
            linear_output += w[j] * input[j];
        }
        linear_output += b;
        return sigmoid(linear_output);
    }

    //构造破损数据集
/*    public void sample_h_given_v(double[] input, double[] sample) {
        for (int i = 0; i < n_out; i++) {
            sample[i] = binomial(1, output(input, W[i], b[i]));
        }
    }*/
}
