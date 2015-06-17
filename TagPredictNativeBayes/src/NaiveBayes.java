import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;


public class NaiveBayes implements Serializable {

    // The Language parameter
    static private double alpha = 0.1;

    HashMap<String, Double> priorProb;
    HashMap<String, HashMap<String, Double>> posteriorProb;
    HashMap<String, Integer> wordCount;
    HashMap<String, Integer> tagCount;

    private static ArrayList<String> cutWords(String text) {
        SnowballStemmer stemmer = new englishStemmer();
        ArrayList<String> words = new ArrayList<String>();
        // using ApacheLucene StandardAnalyzer to cut words
        Reader reader = new StringReader(text);
        Analyzer analyzer = new StandardAnalyzer();
        try {
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
        } catch (IOException e) {
            // I believe this is impossible...
            e.printStackTrace();
        } finally {
            analyzer.close();
        }
        return words;
    }

    public void train(String dataFile) throws IOException {
        wordCount = new HashMap<String, Integer>();
        tagCount = new HashMap<String, Integer>();
        HashMap<String, HashMap<String, Integer>> tagWordCount = new HashMap<String, HashMap<String, Integer>>();

        // Reading Data File
        BufferedReader br = new BufferedReader(new FileReader(dataFile));
        String line;
        int lineNumber = 0;
        while ((line = br.readLine()) != null) {
            if (line.trim().isEmpty()) continue;
            // if it is not an empty line
            lineNumber++;
            String[] split = line.split("#\\$#");
            String description = split[1].replaceAll("[^\\x20-\\x7F]", "_");
            String[] tagList = split[2].split(",");
            ArrayList<String> wordList = cutWords(description);

            // Count words
            for (String word : wordList) {
                if (wordCount.containsKey(word)) {
                    wordCount.replace(word, wordCount.get(word) + 1);
                } else {
                    wordCount.put(word, 1);
                }
            }

            // Count tags
            for (String tag : tagList) {
                if (tagCount.containsKey(tag)) {
                    tagCount.replace(tag, tagCount.get(tag) + 1);
                } else {
                    tagCount.put(tag, 1);
                }
                // Count words having this tag (for Bayes)
                if (tagWordCount.containsKey(tag)) {
                    HashMap<String, Integer> tagWord = tagWordCount.get(tag);
                    for (String word : wordList) {
                        if (tagWord.containsKey(word)) {
                            tagWord.replace(word, tagWord.get(word) + 1);
                        } else {
                            tagWord.put(word, 1);
                        }
                    }
                } else {
                    HashMap<String, Integer> tagWord = new HashMap<String, Integer>();
                    for (String word : wordList) {
                        if (tagWord.containsKey(word)) {
                            tagWord.replace(word, tagWord.get(word) + 1);
                        } else {
                            tagWord.put(word, 1);
                        }
                    }
                    tagWordCount.put(tag, tagWord);
                }
            }
        }
        br.close();

        // Remove words which showing times < 3 or length < 3
        Iterator<HashMap.Entry<String, Integer>> iter = wordCount.entrySet().iterator();
        while (iter.hasNext()) {
            HashMap.Entry<String, Integer> entry = iter.next();
            if (entry.getValue() < 3 || entry.getKey().length() < 3) {
                iter.remove();
            }
        }

        // Remove tags which showing times < 3
        iter = tagCount.entrySet().iterator();
        while (iter.hasNext()) {
            HashMap.Entry<String, Integer> entry = iter.next();
            if (entry.getValue() < 3) {
                iter.remove();
            }
        }

        // Now do the Bayes part. First we calculate prior probability
        priorProb = new HashMap<String, Double>();
        for (HashMap.Entry<String, Integer> tagEntry : tagCount.entrySet()) {
            priorProb.put(tagEntry.getKey(), (double) tagEntry.getValue() / (double) lineNumber);
        }

        // Then we calculate posterior probability
        posteriorProb = new HashMap<String, HashMap<String, Double>>();
        for (String tag : tagCount.keySet()) {
            HashMap<String, Double> oneTagPosterior = new HashMap<String, Double>();
            int totalWordsNum = 0;
            for (Integer n : tagWordCount.get(tag).values()) {
                totalWordsNum += n;
            }
            for (String word : wordCount.keySet()) {
                if (tagWordCount.get(tag).get(word) != null) {
                    double count = (double) tagWordCount.get(tag).get(word);
                    oneTagPosterior.put(word, (count + alpha) / (totalWordsNum + alpha * wordCount.size()));
                } else {
                    oneTagPosterior.put(word, alpha / (totalWordsNum + alpha * wordCount.size()));
                }
            }
            posteriorProb.put(tag, oneTagPosterior);
        }
    }

    public List<String> test(String description) {
        ArrayList<String> wordList = null;
        wordList = cutWords(description);
        HashMap<String, Double> probability = new HashMap<String, Double>();
        for (String tag : tagCount.keySet()) {
            double prob = priorProb.get(tag);
            for (String word : wordList) {
                if (!wordCount.containsKey(word)) continue;
                prob *= posteriorProb.get(tag).get(word);
            }
            probability.put(tag, prob);
        }
        List<HashMap.Entry<String, Double>> list =
                new LinkedList<HashMap.Entry<String, Double>>(probability.entrySet());
        // Sort as large to small
        Collections.sort(list, new Comparator<HashMap.Entry<String, Double>>() {
            @Override
            public int compare(HashMap.Entry<String, Double> o1, HashMap.Entry<String, Double> o2) {
                return -o1.getValue().compareTo(o2.getValue());
            }
        });
        List<String> tagsSorted = new ArrayList<String>(tagCount.size());
        for (HashMap.Entry<String, Double> entry : list) {
            tagsSorted.add(entry.getKey());
        }
        return tagsSorted;
    }
}
