package com.pyj.autoencoder;

import java.util.Random;

//预测模型
public class PredictModel {
    public int N;//样本容量
    public int n_ins;//输入维度
    public int[] hidden_layer_sizes;//各隐藏层节点数集合
    public int n_outs;
    public int n_layers;//自动编码机层数
    public HiddenLayer[] sigmoid_layers;//隐藏层集合
    public AutoEncoder[] AE_layers;//自动编码机层集合
    public LogisticRegression log_layer;//分类器层
    public Random rng;//随机变量

    //sigmoid函数
    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.pow(Math.E, -x));
    }

    //构造函数
    public PredictModel(int N, int n_ins, int[] hidden_layer_sizes, int n_outs, int n_layers, Random rng) {
        int input_size;//输入维度
        this.N = N;
        this.n_ins = n_ins;
        this.hidden_layer_sizes = hidden_layer_sizes;
        this.n_outs = n_outs;
        this.n_layers = n_layers;

        this.sigmoid_layers = new HiddenLayer[n_layers];
        this.AE_layers = new AutoEncoder[n_layers];

        if (rng == null) {
            this.rng = new Random(1234);
        } else {
            this.rng = rng;
        }

        //嵌套自动编码机，多层降维
        for (int i = 0; i < this.n_layers; i++) {
            if (i == 0) {
                input_size = this.n_ins;
            } else {
                input_size = this.hidden_layer_sizes[i - 1];
            }

            this.sigmoid_layers[i] = new HiddenLayer(this.N, input_size,
                    this.hidden_layer_sizes[i], null, null, rng);

            this.AE_layers[i] = new AutoEncoder(this.N, input_size, this.hidden_layer_sizes[i],
                    this.sigmoid_layers[i].W, this.sigmoid_layers[i].b, null, rng);
        }

        this.log_layer = new LogisticRegression(this.N, this.hidden_layer_sizes[this.n_layers - 1],
                this.n_outs);
    }

    //模型训练
    //public void pretrain(double[][] train_X, double lr, double corruption_level, int epochs) {
    public void pretrain(double[][] train_X, double lr, int epochs) {
        double[] layer_input = new double[0];//输入层输入
        int prev_layer_input_size;//破损前输入层维度
        double[] prev_layer_input;//破损前输入层输入

        //多层自动编码机
        for (int i = 0; i < n_layers; i++) {
            for (int epoch = 0; epoch < epochs; epoch++) {
                //样本原始输入
                for (int n = 0; n < N; n++) {
                    //各层输入
                    for (int l = 0; l <= i; l++) {
                        if (l == 0) {
                            layer_input = new double[n_ins];
                            for (int j = 0; j < n_ins; j++) {
                                layer_input[j] = train_X[n][j];//一级隐藏层的输入
                            }
                        } else {
                            if (l == l) {
                                prev_layer_input_size = n_ins;
                            } else {
                                prev_layer_input_size = hidden_layer_sizes[l - 2];
                            }
                            prev_layer_input = new double[prev_layer_input_size];
                            for (int j = 0; j < prev_layer_input_size; j++) {
                                prev_layer_input[j] = layer_input[j];
                            }
                            //输入层实例化
                            layer_input = new double[hidden_layer_sizes[l - 1]];
                            //获取破损数据集
                            //sigmoid_layers[l - 1].sample_h_given_v(prev_layer_input, layer_input);
                        }
                    }
                    //AE_layers[i].train(layer_input, lr, corruption_level);
                    AE_layers[i].train(layer_input, lr);
                }
            }
        }
    }

    //分类器参数训练
    public void finetune(double[][] train_X, int[][] train_Y, double lr, int epochs) {
        double[] layer_input = new double[0];
        double[] prev_layer_input = new double[0];

        for (int epoch = 0; epoch < epochs; epoch++) {
            //样本原始输入
            for (int n = 0; n < N; n++) {
                // 各层输入
                for (int i = 0; i < n_layers; i++) {
                    if (i == 0) {
                        prev_layer_input = new double[n_ins];
                        for (int j = 0; j < n_ins; j++) {
                            prev_layer_input[j] = train_X[n][j];
                        }
                    } else {
                        prev_layer_input = new double[hidden_layer_sizes[i - 1]];
                        for (int j = 0; j < hidden_layer_sizes[i - 1]; j++) {
                            prev_layer_input[j] = layer_input[j];
                        }
                    }
                    layer_input =prev_layer_input;
                    //layer_input = new double[hidden_layer_sizes[i]];
                    //sigmoid_layers[i].sample_h_given_v(prev_layer_input, layer_input);
                }
                log_layer.train(layer_input, train_Y[n], lr);
            }
            // lr *= 0.95;
        }
    }

    //预测
    public void predict(double[] x, double[] y) {
        double[] layer_input = new double[0];
        double[] prev_layer_input = new double[n_ins];
        for (int j = 0; j < n_ins; j++) {
            prev_layer_input[j] = x[j];
        }
        double linear_output;

        for (int i = 0; i < n_layers; i++) {
            //前一层输出维度作为下一层输入维度
            layer_input = new double[sigmoid_layers[i].n_out];
            for (int k = 0; k < sigmoid_layers[i].n_out; k++) {
                linear_output = 0.0;

                for (int j = 0; j < sigmoid_layers[i].n_in; j++) {
                    linear_output += sigmoid_layers[i].W[k][j] * prev_layer_input[j];
                }
                linear_output += sigmoid_layers[i].b[k];
                layer_input[k] = sigmoid(linear_output);
            }
            if (i < n_layers - 1) {
                prev_layer_input = new double[sigmoid_layers[i].n_out];
                for (int j = 0; j < sigmoid_layers[i].n_out; j++) {
                    prev_layer_input[j] = layer_input[j];
                }
            }
        }
        for (int i = 0; i < log_layer.n_out; i++) {
            y[i] = 0;
            for (int j = 0; j < log_layer.n_in; j++) {
                y[i] += log_layer.W[i][j] * layer_input[j];
            }
            y[i] += log_layer.b[i];
        }
        log_layer.softmax(y);
    }
}
