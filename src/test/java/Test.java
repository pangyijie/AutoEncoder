
import com.pyj.autoencoder.DataSet;
import com.pyj.autoencoder.PredictModel;

import java.io.*;
import java.util.Random;

public class Test {

    public static void main(String[] args) throws Exception {
        Random rng = new Random(123);

        double pretrain_lr = 0.1;
        //double corruption_level = 0.3;
        int pretraining_epochs = 1000;
        double finetune_lr = 0.1;
        int finetune_epochs = 500;

        int train_N = 30;
        int test_N = 10;
        int n_ins = 13;
        int n_outs = 3;
        int[] hidden_layer_sizes = {6, 6};
        int n_layers = hidden_layer_sizes.length;

        // 训练数据
        DataSet ds = new DataSet();
        ds.getTrainData();
        double[][] train_X = ds.train_X;
        int[][] train_Y = ds.train_Y;

        PredictModel pm = new PredictModel(train_N, n_ins, hidden_layer_sizes, n_outs, n_layers, rng);

        //pm.pretrain(train_X, pretrain_lr, corruption_level, pretraining_epochs);
        pm.pretrain(train_X, pretrain_lr, pretraining_epochs);

        pm.finetune(train_X, train_Y, finetune_lr, finetune_epochs);

        ds.getTestData();
        double[][] test_X = ds.test_X;

        double[][] test_Y = new double[test_N][n_outs];

        // test
        FileWriter output = new FileWriter("D://result.txt");
        BufferedWriter bf = new BufferedWriter(output);
        for (int i = 0; i < test_N; i++) {
            pm.predict(test_X[i], test_Y[i]);
            for (int j = 0; j < n_outs; j++) {
                bf.write(test_Y[i][j] + " ");
            }
            bf.newLine();
        }
        bf.close();
/*        DataSet ds = new DataSet();
        double[][] train_X;
        int[][] train_Y;
        double[][] test_X;

        ds.getTrainData();
        ds.getTestData();
        train_X = ds.train_X;
        train_Y = ds.train_Y;
        test_X = ds.test_X;

        //打印train_X
        for (int i = 0; i < 10; i++) {
            StringBuffer str = new StringBuffer();
            for (int j = 0; j < 13; j++) {
                str.append(train_X[i][j]).append(" ");
            }
            System.out.println(str);
        }
        //打印train_Y
        for (int i = 0; i < 10; i++) {
            StringBuffer str = new StringBuffer();
            for (int j = 0; j < 3; j++) {
                str.append(train_Y[i][j]).append(" ");
            }
            System.out.println(str);
        }
        //打印test_X
        for (int i=0;i<4;i++) {
            StringBuffer str = new StringBuffer();
            for (int j = 0; j < 13; j++) {
                str.append(test_X[i][j]).append(" ");
            }
            System.out.println(str);
        }*/
    }
}
