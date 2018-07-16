package com.pyj.autoencoder;

import java.util.Random;

//自动编码机
public class AutoEncoder {
    public int N;//样本容量
    public int n_visible;//原特征维度
    public int n_hidden;//隐藏层节点数
    public double[][] W;//权值矩阵
    public double[] hbias;//编码网络偏置向量
    public double[] vbias;//解码网络偏置向量
    public Random rng;

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
    public AutoEncoder(int N, int n_visible, int n_hidden, double[][] W,
                       double[] hbias, double[] vbias, Random rng) {
        this.N = N;
        this.n_visible = n_visible;
        this.n_hidden = n_hidden;
        //若随机数为null,则初始化赋值
        if (rng == null) {
            this.rng = new Random(1234);
        } else {
            this.rng = rng;
        }
        //若权重矩阵为null,则生成随机数矩阵
        if (W == null) {
            this.W = new double[this.n_hidden][this.n_visible];
            double a = 1.0 / this.n_visible;
            for (int i = 0; i < this.n_hidden; i++) {
                for (int j = 0; j < this.n_visible; j++) {
                    this.W[i][j] = uniform(-a, a);
                }
            }
        } else {
            this.W = W;
        }
        //若编码网络偏置向量为null，则初始化赋值为0
        if (hbias == null) {
            this.hbias = new double[this.n_hidden];
            for (int i = 0; i < this.n_hidden; i++) {
                this.hbias[i] = 0;
            }
        } else {
            this.hbias = hbias;
        }
        //若解码网络偏置向量为null，则初始化赋值为0
        if (vbias == null) {
            this.vbias = new double[this.n_visible];
            for (int i = 0; i < this.n_visible; i++) {
                this.vbias[i] = 0;
            }
        } else {
            this.vbias = vbias;
        }
    }

/*    public void get_corrupted_input(double[] x, double[] tilde_x, double p) {
        for (int i = 0; i < n_visible; i++) {
            if (x[i] == 0) {
                tilde_x[i] = 0;
            } else {
                tilde_x[i] = binomial(1, p);
            }
        }
    }*/

    //编码
    public void get_hidden_values(double[] x, double[] y) {
        for (int i = 0; i < n_hidden; i++) {
            y[i] = 0;
            for (int j = 0; j < n_visible; j++) {
                y[i] += W[i][j] * x[j];
            }
            y[i] += hbias[i];
            y[i] = sigmoid(y[i]);
        }
    }

    //解码
    public void get_reconstructed_input(double[] y, double[] z) {
        for (int i = 0; i < n_visible; i++) {
            z[i] = 0;
            for (int j = 0; j < n_hidden; j++) {
                z[i] += W[j][i] * y[j];
            }
            z[i] += vbias[i];
            z[i] = sigmoid(z[i]);
        }
    }

    //训练模型
    //public void train(double[] x, double lr, double corruption_level) {
    public void train(double[] x, double lr) {
        //double[] tilde_x = new double[n_visible];
        double[] y = new double[n_hidden];
        double[] z = new double[n_visible];

        double[] L_vbias = new double[n_visible];
        double[] L_hbias = new double[n_hidden];

        //double p = 1 - corruption_level;
        //get_corrupted_input(x, tilde_x, p);
        //get_hidden_values(tilde_x, y);
        get_hidden_values(x, y);
        get_reconstructed_input(y, z);

        //更新解码偏置向量
        for (int i = 0; i < n_visible; i++) {
            L_vbias[i] = x[i] - z[i];
            vbias[i] += lr * L_vbias[i] / N;
        }

        //更新编码偏置向量
        for (int i = 0; i < n_hidden; i++) {
            L_hbias[i] = 0;
            for (int j = 0; j < n_visible; j++) {
                L_hbias[i] += W[i][j] * L_vbias[j];
            }
            L_hbias[i] *= y[i] * (1 - y[i]);
            hbias[i] += lr * L_hbias[i] / N;
        }

        //更新权值矩阵
        for (int i = 0; i < n_hidden; i++) {
            for (int j = 0; j < n_visible; j++) {
                //W[i][j] += lr * (L_hbias[i] * tilde_x[j] + L_vbias[j] * y[i]) / N;
                W[i][j] += lr * (L_hbias[i] * x[j] + L_vbias[j] * y[i]) / N;
            }
        }
    }

    //数据重构
    public void reconstruct(double[] x, double[] z) {
        double[] y = new double[n_hidden];
        get_hidden_values(x, y);
        get_reconstructed_input(y, z);
    }
}
