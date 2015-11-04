import java.io.*;
import java.util.*;
//import edu.stanford.nlp.dcoref.CorefChain;
//import edu.stanford.nlp.dcoref.CorefCoreAnnotations;
//import edu.stanford.nlp.parser.Parser;
import edu.stanford.nlp.parser.ui.TreeJPanel;
//import edu.stanford.nlp.io.*;
//import java.awt.Graphics;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.sentiment.CollapseUnaryTransformer;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.*;
import javax.swing.*;


/**
    Please DO NOT delete the commented functions, I use them for debugging
    - Ali
**/

/** This class demonstrates building and using a Stanford CoreNLP pipeline. */
public class MakeTrees {
    public static void gain(String[] args) throws Exception{
        PrintWriter writer = new PrintWriter("check.txt", "UTF-8");

        writer.println("1");
        writer.println("2");
        for(int i=0; i<10; i++)
            writer.println(Integer.toString(i));

        writer.close();
    }

    public static void main(String[] args) throws Exception{
        BufferedReader br;
        String line;
        String delimiter = ",(?=([^\"]*\"[^\"]*\")*[^\"]*$)";
        String inputFilename = "wiki_quartiles_cleaned.csv";
        String outputFilename = "WikiTreebanks.txt";

        try {
            br = new BufferedReader(new FileReader(inputFilename));
            // Output files
            PrintWriter out;
            out = new PrintWriter(outputFilename);
            PrintWriter anomalies = new PrintWriter("anomalies.txt");

            // Create a CoreNLP pipeline. This line just builds the default pipeline.
            Properties props = new Properties();
            props.put("annotators", "tokenize, ssplit, parse, sentiment");
            StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
            Annotation annotation;
            List<CoreMap> sentences;
            Tree tree;
            CollapseUnaryTransformer collapser = new CollapseUnaryTransformer();
            TreePrint tp = new TreePrint("oneline");

            // Skip first line
            //line = br.readLine();

            // Read all other lines
            while((line = br.readLine()) != null){
                String[] wiki = line.split(delimiter);
                // wiki[0] = id, wiki[1] = sentence, wiki[2] = annotation
                wiki[1] = wiki[1].replaceAll("(^\")|(\"$)", "");
                //System.out.println(wiki[1]);
                annotation = new Annotation(wiki[1]);

                // Annotate the sentence
                pipeline.annotate(annotation);

                // Now every line is two sentences, so there will be two trees
                // Fetch the trees
                sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);

                if(sentences.size() == 2) {
                    for (CoreMap sentence : sentences) {
                        // Pick first sentence
                        //CoreMap sentence = sentences.get(0);
                        tree = sentence.get(TreeCoreAnnotations.BinarizedTreeAnnotation.class);
                        // Collapse dummy terminals at the tree end
                        tree = collapser.transformTree(tree);
                        out.print(wiki[0] + " ");
                        tp.printTree(tree, out);
                    }
                }
                else{
                    System.out.println("Anomaly occured at id " + wiki[0]);
                    anomalies.println(wiki[0]);
                }
            }

            // Close the file
            br.close();
            anomalies.close();
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        catch (IOException e){
            e.printStackTrace();
        }
        catch (IndexOutOfBoundsException e){
            e.printStackTrace();
            System.out.println("Error aaya hai.");
        }
    }

    /*
    public static void pain(String[] args){
        Properties props = new Properties();
        props.put("annotators", "tokenize, ssplit, parse, sentiment");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        Annotation annotation;
        CoreMap sentence;
        Tree tree;

        annotation = new Annotation("How many times do I have to revert you before you stop?");

        // Annotate the sentence
        pipeline.annotate(annotation);

        // Now every line is two sentences, so there will be two trees
        // Fetch the trees
        sentence = annotation.get(CoreAnnotations.SentencesAnnotation.class).get(0);
        tree = sentence.get(TreeCoreAnnotations.BinarizedTreeAnnotation.class);

        // Collapse the tree
        CollapseUnaryTransformer transformer = new CollapseUnaryTransformer();
        tree = transformer.transformTree(tree);

        JFrame mainFrame = new JFrame("Binary Tree Examples");
        mainFrame.setSize(400,400);
        TreeJPanel treePanel = new TreeJPanel();
        treePanel.setTree(tree);
        mainFrame.add(treePanel);
        mainFrame.setVisible(true);
    }
    */

}

/* Removed GUI stuff
    JFrame mainFrame = new JFrame("Java Swing Examples");
    mainFrame.setSize(400,400);
    TreeJPanel treePanel = new TreeJPanel();

    treePanel.setTree(binaryTree);
    mainFrame.add(treePanel);
    mainFrame.setVisible(true);
 */
