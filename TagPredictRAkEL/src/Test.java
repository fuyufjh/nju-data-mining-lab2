import mulan.classifier.MultiLabelOutput;
import mulan.classifier.meta.RAkEL;
import mulan.data.MultiLabelInstances;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;
import weka.core.Instance;
import weka.core.SerializationHelper;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class Test {

    // Params
    static String inputFile = "test.data";
    static String arffFile = "test.arff";
    static String xmlFilename = "tag_predict.xml";

    static HashMap<Integer, HashMap<String, Float>> tfMap;
    static HashMap<String, Float> idfMap;
    static HashMap<Integer, Set<String>> tagsMap;
    static ArrayList<String> wordList;
    static ArrayList<String> tagList;

    public static void main(String[] args) throws Exception {

        // Translate into ARFF file
        ObjectInputStream ois = new ObjectInputStream(new FileInputStream("TFIDF"));
        idfMap = (HashMap<String, Float>) ois.readObject();
        wordList = (ArrayList<String>) ois.readObject();
        tagList = (ArrayList<String>) ois.readObject();
        translate(inputFile, arffFile);

        // Load model
        System.out.print("Reading Model...");
        RAkEL model = (RAkEL) SerializationHelper.read("Model");
        System.out.println("Done");

        BufferedWriter bw = new BufferedWriter(new FileWriter("result.txt"));
        MultiLabelInstances testData = new MultiLabelInstances(arffFile, xmlFilename);
        String[] labels = testData.getLabelNames();
        int numInstances = testData.getNumInstances();
        for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
            Instance instance = testData.getDataSet().instance(instanceIndex);
            MultiLabelOutput output = model.makePrediction(instance);
            if (output.hasBipartition()) {
                boolean[] bipartion = output.getBipartition();
                bw.write(Integer.toString(instanceIndex + 1) + "#$#");
                for (int i = 0; i < bipartion.length; i++) {
                    if (bipartion[i]) bw.write(labels[i].replace("_", " ") + ",");
                }
                bw.write("\r\n");
            }
        }
        bw.close();
    }

    static SnowballStemmer stemmer = new englishStemmer();

    static ArrayList<String> cutWords(String text) throws IOException {
        ArrayList<String> words = new ArrayList<String>();
        // using ApacheLucene StandardAnalyzer to cut words
        Reader reader = new StringReader(text);
        Analyzer analyzer = new StandardAnalyzer();
        TokenStream tokenStream = analyzer.tokenStream("", reader);
        CharTermAttribute term = tokenStream.getAttribute(CharTermAttribute.class);
        tokenStream.reset();
        while (tokenStream.incrementToken()) {
            String word = term.toString();
            // if it is number, just skip
            if (word.matches("^\\d+(\\.\\d+)?$")) continue;
            // Stemming
            stemmer.setCurrent(word);
            if (stemmer.stem()) {
                words.add(stemmer.getCurrent());
            } else {
                words.add(word);
            }
        }
        analyzer.close();
        return words;
    }

    public static void translate(String inputFile, String outputFile) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(inputFile));
        BufferedWriter bw = new BufferedWriter(new FileWriter(outputFile));

        // ARFF Headers
        bw.write("@relation tagpred\n\n");

        String line;
        int lineNumber = 0;
        tfMap = new HashMap<Integer, HashMap<String, Float>>();
        tagsMap = new HashMap<Integer, Set<String>>();
        while ((line = br.readLine()) != null) {
            lineNumber++;
            String[] split = line.split("#\\$#");
            String description = split[1].replaceAll("[^\\x20-\\x7F]", "_");

            ArrayList<String> wordList = cutWords(description);

            HashMap<String, Integer> statistics = new HashMap<String, Integer>();
            for (String word : wordList) {
                // Count in only this file (for TF)
                if (statistics.get(word) == null) {
                    statistics.put(word, 1);
                } else {
                    statistics.replace(word, statistics.get(word) + 1);
                }
            }
            // Calculate & save TF value
            HashMap<String, Float> termFrequency = new HashMap<String, Float>();
            for (Map.Entry<String, Integer> entry : statistics.entrySet()) {
                termFrequency.put(entry.getKey(), (float) entry.getValue() / wordList.size());
            }
            tfMap.put(lineNumber, termFrequency);
        }
        // ARFF Attributions declaration
        for (int i = 0; i < wordList.size(); i++) {
            bw.write(String.format("@attribute tfidf_%s numeric\n", wordList.get(i)));
        }
        for (int i = 0; i < tagList.size(); i++) {
            bw.write(String.format("@attribute %s {0,1}\n", tagList.get(i).replace(" ", "_")));
        }
        bw.write("\n@data\n");

        // Calculate & save IF-IDF value to file
        for (int i = 1; i <= lineNumber; i++) {
            // Write IF-IDF features
            for (int j = 0; j < wordList.size(); j++) {
                String word = wordList.get(j);
                if (j > 0) bw.write(",");
                if (tfMap.get(i).containsKey(word)) {
                    bw.write(String.format("%f", tfMap.get(i).get(word) * idfMap.get(word)));
                } else {
                    bw.write("0");
                }
            }
            // Write labels by "?"
            for (int j = 0; j < tagList.size(); j++) {
                bw.write(",?");
            }
            bw.write("\n");
        }
        bw.close();
        br.close();
    }
}
