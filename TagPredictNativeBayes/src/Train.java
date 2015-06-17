import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

public class Train {

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        NaiveBayes model = new NaiveBayes();
        System.out.print("Training...");
        model.train("train.data");
        System.out.println("Done");

        // Save model
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("model"));
        oos.writeObject(model);
        oos.close();

        System.out.println("Model saved");
    }
}
