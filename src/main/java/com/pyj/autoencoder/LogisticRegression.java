package com.pyj.autoencoder;

//分类器
public class LogisticRegression {
    public int N;//样本容量
    public int n_in;//输入维度，即样本大小
    public int n_out;//输出的类别数
    public double[][] W;//权值矩阵
    public double[] b;//偏置向量

    //构造函数
    public LogisticRegression(int N, int n_in, int n_out) {
        this.N = N;
        this.n_in = n_in;
        this.n_out = n_out;

        W = new double[this.n_out][this.n_in];
        b = new double[this.n_out];
    }

    //softmax分类器
    public void softmax(double[] x) {
        double max = 0.0;
        double sum = 0.0;

        for (int i = 0; i < n_out; i++) {
            if (max < x[i]) {
                max = x[i];
            }
        }

        for (int i = 0; i < n_out; i++) {
            x[i] = Math.exp(x[i] - max);
            sum += x[i];
        }

        for (int i = 0; i < n_out; i++) {
            x[i] /= sum;
        }
    }

    //参数训练
    public void train(double[] x, int[] y, double lr) {
        double[] p_y_given_x = new double[n_out];//每一行代表每一个样本被估计为各类别的概率
        double[] dy = new double[n_out];

        for (int i = 0; i < n_out; i++) {
            p_y_given_x[i] = 0;
            for (int j = 0; j < n_in; j++) {
                p_y_given_x[i] += W[i][j] * x[j];
            }
            p_y_given_x[i] += b[i];
        }
        softmax(p_y_given_x);

        for (int i = 0; i < n_out; i++) {
            dy[i] = y[i] - p_y_given_x[i];
            for (int j = 0; j < n_in; j++) {
                W[i][j] += lr * dy[i] * x[j] / N;
            }
            b[i] += lr * dy[i] / N;
        }
    }

    //预测
    public void predict(double[] x, double[] y) {
        for (int i = 0; i < n_out; i++) {
            y[i] = 0;
            for (int j = 0; j < n_in; j++) {
                y[i] += W[i][j] * x[j];
            }
            y[i] += b[i];
        }
        softmax(y);
    }
}
