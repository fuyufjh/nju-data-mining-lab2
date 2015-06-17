import java.io.*;
import java.util.List;

public class Test {

    // How many labels to predict
    static private int topN = 4;

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        // Open model
        System.out.print("Loading model...");
        ObjectInputStream ois = new ObjectInputStream(new FileInputStream("model"));
        NaiveBayes model = (NaiveBayes) ois.readObject();
        ois.close();
        System.out.println("Done");

        // read the test data
        BufferedReader br = new BufferedReader(new FileReader("test.data"));
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = br.readLine()) != null) {
            String[] split = line.split("#\\$#");
            String no = split[0];
            String description = split[1].replaceAll("[^\\x20-\\x7F]", "_");
            String[] tags = split[2].split(",");
            List<String> tagsPredicted = model.test(description).subList(0, topN);

            // Output predicted labels
            sb.append(no).append("#$#");
            for (int i = 0; i < topN; i++) {
                sb.append(tagsPredicted.get(i)).append(i == topN - 1 ? "\r\n" : ",");
            }
        }
        BufferedWriter bw = new BufferedWriter(new FileWriter("result.txt"));
        bw.write(sb.toString());
        bw.close();
    }
}
