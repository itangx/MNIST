package co.kulwadee.int492.lect05;

import org.apache.commons.io.FilenameUtils;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;

import javax.swing.*;
import java.io.File;

public class MnistClassifierLoadModel {
    private static Logger log = LoggerFactory.getLogger(MnistClassifierLoadModel.class);

    public static final String DATA_PATH =  FilenameUtils.concat(System.getProperty("user.home"), "dl4j_mnist/");

    public static String imageFileChose() {
        JFileChooser fc = new JFileChooser();
        int ret = fc.showOpenDialog(null);
        if (ret == JFileChooser.APPROVE_OPTION) {
            File file = fc.getSelectedFile();
            String filename = file.getAbsolutePath();
            return filename;
        } else {
            return null;
        }
    }

    public static void main(String[] args) throws Exception {
        // Input images properties
        int height = 28;
        int width = 28;
        int channels = 1;

        String imageFileName = imageFileChose();

        // Load a trained network
        File locationToSave = new File(DATA_PATH + "/trained_mnist_model.zip");
        if (locationToSave.exists()) {
            log.info("Saved model found at: " + locationToSave.getAbsolutePath());
        } else {
            log.error("No saved model found!");
            log.error(locationToSave.getAbsolutePath());
            System.exit(0);
        }

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        log.info("TEST SELECTED IMAGE AGAINST SAVED NETWORK.");

        File imageFile = new File(imageFileName);

        // convert raw image pixels to numerical matrix
        NativeImageLoader loader = new NativeImageLoader(height, width, channels);

        // Put the image into an INDarray
        INDArray image = loader.asMatrix(imageFile);

        // Normalize the image
        // 0-255 pixel values => 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);

        // Predict the digit by passing the image into the neural network
        INDArray output = model.output(image);

        log.info("The image file chosen was " + imageFileName);
        log.info("The neural network prediction (list of probabilities per label)");
        log.info(output.toString());

        INDArray idxOfMaxInEachRow = Nd4j.getExecutioner().exec(new IAMax(output),1);
        log.info("Predicted Digit: " + Math.round(Float.parseFloat(idxOfMaxInEachRow.toString())));
        log.info("Actual Digit (loaded from file " + imageFileName + ") is " + imageFile.getParentFile().getName());

    }
}
