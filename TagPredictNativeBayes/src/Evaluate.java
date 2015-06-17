import java.io.*;
import java.lang.reflect.Array;
import java.util.*;

public class Evaluate {

    // How many labels to predict
    static private int topN = 4;

    static private int nTags = 354;

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        // Open model
        System.out.print("Loading model...");
        ObjectInputStream ois = new ObjectInputStream(new FileInputStream("model"));
        NaiveBayes model = (NaiveBayes) ois.readObject();
        ois.close();
        System.out.println("Done");

        Map<String, Integer> nTruePositive = new HashMap<String, Integer>(nTags);
        Map<String, Integer> nFalsePositive = new HashMap<String, Integer>(nTags);
        Map<String, Integer> nFalseNegative = new HashMap<String, Integer>(nTags);
        double hammingLoss = 0;

        // read the test data
        BufferedReader br = new BufferedReader(new FileReader("test.data"));
        int nCases = 0;
        String line;
        while ((line = br.readLine()) != null) {
            line = line.replaceAll("[^\\x20-\\x7F]", "_");
            nCases++;
            String[] split = line.split("#\\$#");
            String no = split[0];
            String description = split[1];
            List<String> tagsCorrect = Arrays.asList(split[2].split(","));
            List<String> tagsPredicted = model.test(description).subList(0, topN);

            // Print predicted labels
            System.out.println(String.format("%s: %s", no, tagsPredicted.toString()));

            // For calculating Micro & Macro F1
            int nInter = 0;
            for (String tag : tagsCorrect) {
                if (tagsPredicted.contains(tag)) {
                    nInter++;
                    int origin = nTruePositive.containsKey(tag) ? nTruePositive.get(tag) : 0;
                    nTruePositive.put(tag, origin + 1);
                } else {
                    int origin = nFalseNegative.containsKey(tag) ? nFalseNegative.get(tag) : 0;
                    nFalseNegative.put(tag, origin + 1);
                }
            }
            for (String tag : tagsPredicted) {
                if (!tagsCorrect.contains(tag)) {
                    int origin = nFalsePositive.containsKey(tag) ? nFalsePositive.get(tag) : 0;
                    nFalsePositive.put(tag, origin + 1);
                }
            }

            // For calculating Hamming Loss
            hammingLoss += (double) (tagsCorrect.size() + tagsPredicted.size() - nInter * 2) / (double) nTags;
        }

        double sumTP = sumUpValue(nTruePositive);
        double sumFP = sumUpValue(nFalsePositive);
        double sumFN = sumUpValue(nFalseNegative);

        double microF1 = calculateF1(sumTP, sumFP, sumFN);
        double macroF1 = 0;
        for (String tag : nTruePositive.keySet()) {
            double nTP = nTruePositive.get(tag) != null ? nTruePositive.get(tag) : 0;
            double nFP = nFalsePositive.get(tag) != null ? nFalsePositive.get(tag) : 0;
            double nFN = nFalseNegative.get(tag) != null ? nFalseNegative.get(tag) : 0;
            macroF1 += calculateF1(nTP, nFP, nFN);
        }
        macroF1 /= nTags;
        hammingLoss /= nCases;
        System.out.println(String.format("HammingLoss = %f\nMacro-F1 = %f\nMicro-F1 = %f",
                hammingLoss, macroF1, microF1));
    }

    private static int sumUpValue(Map<String, Integer> m) {
        int sum = 0;
        for (int i : m.values()) {
            sum += i;
        }
        return sum;
    }

    private static double calculateF1(double nTP, double nFP, double nFN) {
        double precision = nTP / (nTP + nFP);
        double recall = nTP / (nTP + nFN);
        return 2.0 * (precision * recall) / (precision + recall);
    }
}
