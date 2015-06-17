import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;


public class Translate {
	
	static Pattern regexWord = Pattern.compile("[\\w]+");
	static Pattern regexDigit = Pattern.compile("[0-9]+");

	static String descriptions;
	static HashMap<Integer, HashMap<String, Float>> tfMap;
	static HashMap<String, Float> idfMap;
	static HashMap<Integer, HashMap<String, Float>> tfidfMap;
	static HashMap<Integer, Set<String>> tagsMap;
	static ArrayList<String> wordList;
	static ArrayList<String> tagList;
	
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
	
	public static void translate(FileInputStream inputStream, 
			FileOutputStream outputStream) throws IOException {
		InputStreamReader inputStreamReader = new InputStreamReader(inputStream);
        BufferedReader br = new BufferedReader(inputStreamReader); 
		OutputStreamWriter outputStreamWriter = new OutputStreamWriter(outputStream);
		BufferedWriter bw = new BufferedWriter(outputStreamWriter);
		
		// ARFF Headers
		bw.write("@relation tagpred\n\n");
		
		String line;
		int lineNumber = 0;
		tfMap = new HashMap<Integer, HashMap<String, Float>>();
		idfMap = new HashMap<String, Float>();
		tagsMap = new HashMap<Integer, Set<String>>();
		HashMap<String, Integer> wordCount = new HashMap<String, Integer>();
		while ((line = br.readLine()) != null) {
			lineNumber++;
			String[] splited = line.split("#\\$#");
			String description = splited[1].replaceAll("[^\\x20-\\x7F]", "_");
			String taglist = splited[2];
			String[] tags = taglist.split(",");
			tagsMap.put(lineNumber, new HashSet<String>(Arrays.asList(tags)));
			
			ArrayList<String> wordList = cutWords(description);
			
			HashMap<String, Integer> statistics = new HashMap<String, Integer>();
			for (String word: wordList) {
				// Count in only this file (for TF)
				if (statistics.get(word) == null) {
					statistics.put(word, 1);
				} else {
					statistics.replace(word, statistics.get(word) + 1);
				}
			}
			// Count # of files that this word is shown (for IDF) 
			for (Map.Entry<String, Integer> entry: statistics.entrySet()) {
				String word = entry.getKey();
				if (wordCount.get(word) == null) {
					wordCount.put(word, 1);
				} else {
					wordCount.replace(word, wordCount.get(word) + 1);
				}
			}
			// Calculate & save TF value
			HashMap<String, Float> termFrequency = new HashMap<String, Float>();
			for (Map.Entry<String, Integer> entry: statistics.entrySet()) {
				termFrequency.put(entry.getKey().toString(), (float)entry.getValue() / wordList.size());
			}
			tfMap.put(lineNumber, termFrequency);
		}
		// Remove words which showing times < 3 or length <= 2
		Iterator<Map.Entry<String, Integer>> iter = wordCount.entrySet().iterator();
		while (iter.hasNext()) {
			Map.Entry<String, Integer> entry = iter.next();
			if (entry.getValue() < 3 || entry.getKey().length() <= 2) {
				iter.remove();
			}
		}
		// Calculate & save IDF value
		wordList = new ArrayList<String>();
		for (Map.Entry<String, Integer> entry: wordCount.entrySet()) {
			idfMap.put(entry.getKey(), (float) Math.log((float) lineNumber / entry.getValue()));
			wordList.add(entry.getKey());
		}
		// ARFF Attributions declaration
		for (int i = 0; i < wordList.size(); i++) {
			bw.write(String.format("@attribute tfidf_%s numeric\n", wordList.get(i)));
		}
		for (int i = 0; i < tagList.size(); i++) {
			bw.write(String.format("@attribute %s {0,1}\n", tagList.get(i).replace(' ', '_')));
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
			// Write labels by 0/1
			for (int j = 0; j < tagList.size(); j++) {
				if (tagsMap.get(i).contains(tagList.get(j))) {
					bw.write(",1");
				} else {
					bw.write(",0");
				}
				
			}
			bw.write("\n");
		}
		bw.close();
		br.close();
	}
	
	public static void generateTagList(FileInputStream tagFile) throws IOException {
		InputStreamReader inputStreamReader = new InputStreamReader(tagFile);
        BufferedReader br = new BufferedReader(inputStreamReader);
        String line;
        tagList = new ArrayList<String>();
        while ((line = br.readLine()) != null) {
        	if (line.trim().isEmpty()) continue;
        	tagList.add(line);
        }
        br.close();
	}
	
	public static void main(String[] args) throws FileNotFoundException, IOException {
		
		File inputFile = new File("train.data");
		File tagFile = new File("AllTags.txt");
		File outputFile = new File("tag_predict.arff");
		File xmlOutputFile = new File("tag_predict.xml");
		
		outputFile.createNewFile();
		FileInputStream inputStram = new FileInputStream(inputFile);
		FileInputStream tagInputStram = new FileInputStream(tagFile);
		FileOutputStream outputStream = new FileOutputStream(outputFile);
		FileOutputStream xmlOutputStream = new FileOutputStream(xmlOutputFile);
		
		generateTagList(tagInputStram);
		translate(inputStram, outputStream);
		
		inputStram.close();
		tagInputStram.close();
		outputStream.close();
		
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(xmlOutputStream));
		bw.write("<labels xmlns=\"http://mulan.sourceforge.net/labels\">\n");
		for (int j = 0; j < tagList.size(); j++) {
			String tag = tagList.get(j);
			bw.write(String.format("<label name=\"%s\"></label>\n", tag.replace(" ", "_")));
		}
		bw.write("</labels>\n");
		bw.close();
		xmlOutputStream.close();
		
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("TFIDF"));
		oos.writeObject(idfMap);
		oos.writeObject(wordList);
		oos.writeObject(tagList);
		oos.close();
	}

}
