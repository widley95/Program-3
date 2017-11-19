import java.io.File;

import weka.classifiers.Classifier;
import weka.core.*;

public class LogisticPerceptron implements weka.classifiers.Classifier{
    // Input file
    private File file;

    // The number of iterations
    private int numEpochs;

    // Bias for all instances of perceptron
    private int bias;

    // Learning rate
    private double learningRate;
    
    // Logical squashing parameter
    private double squashingParameter;

    // The weights for each perceptron
    private double weights[];

    public LogisticPerceptron(String[] options) throws Exception{
        // Print the header for the output
		System.out.println("\nUniversity of Central Florida ");
		System.out.println("CAP4630 Artificial Intelligence - Fall 2016");
        System.out.println("Perceptron Classifier by Julian Quitian \n");
        
        // Process Arguments
        this.file = new File(options[0]);
        this.numEpochs = Integer.parseInt(options[1]);
        this.learningRate = Double.parseDouble(options[2]);
        this.squashingParameter = Double.parseDouble(options[3]);
        this.bias = 1;
    }

   /**
    * Builds the ensemble of perceptrons.
    *
    * @exception Exception if something goes wrong during building
    */
    public void buildClassifier(Instances data) throws Exception{

        // Make sure there are instances in the set. Return if otherwise.
		if(data.numInstances() == 0){
			return;
        }
        
        // Make sure there are no more than two output classes available
        if(data.numClasses() > 2){
            return;
        }
        
        // Get number of attributes by looking at first instance in dataset
        int numAttributes = data.firstInstance().numAttributes();

        // Set weight vector size. Note: All initialized to 0.0, as required
        weights = new double[numAttributes];

        // Run as many epochs as indicated by user
        for(int epochCounter = 0; epochCounter < this.numEpochs; epochCounter++){

            System.out.print("Epoch " + epochCounter + ": ");
            
            // Go through each attribute in current epoch
            for(int atrCounter = 0; atrCounter < numAttributes; atrCounter++){

                // Current instance/epoch
                Instance currentEpoch = data.instance(epochCounter);

                // Number of attributes in epoch
                int instanceAtrCount = currentEpoch.numAttributes();
                
                // Perform summation of all attribute * weight values
                double sum = 0.0;
                for(int currentAtr = 0; currentAtr < instanceAtrCount; currentAtr++){
                    sum += currentEpoch.attribute(currentAtr).weight() * currentEpoch.value(currentAtr);
                }

                // Account for bias weight
                sum += currentEpoch.attribute(instanceAtrCount - 1).weight() * this.bias;

                // This is the result of the Perceptron. If output does not match expected output, weights will be updated.
                int output = sign(sum);

                // Expected output is last value of each attribute; map +1 to distribution for class 0, and -1 for class 1.
                int expectedOutput = (int)((currentEpoch.value(currentEpoch.attribute(instanceAtrCount - 1)) == 0) ? 1 : -1);

                if(output != expectedOutput){
                    System.out.println("Must train");
                }else{
                    System.out.print("1");
                }
            }
            // Move on to next epoch
			System.out.println();
        }
        
    }

    // Predicts +1 or -1 value for the classification of a sample instance.
    public int predict(Instance instance){
        // Holds all attribute values
        double[] inputs = new double[instance.numAttributes() - 1];

        // Fill input vector with corresponding attribute values
        for(int i = 0; i < inputs.length; i++){
            inputs[i] = instance.value(i);
        }

        double sum = 0;
        // Run initial prediction function, which excludes bias
        for(int i = 0; i < weights.length - 1; i++){
            sum += inputs[i] * weights[i];
        }

        // Account for bias by adding it's corresponding weight to the current sum
        sum += instance.attribute(instance.numAttributes() - 1).weight() * this.bias;

        int output = sign(sum);
        return output; 
    }

    // Empty concrete definition of getCapabilities() as required by the implementation
    public Capabilities getCapabilities(){
        return null;
    };


    // Empty concrete definition of classifyInstance() as required by the implementation
    public double classifyInstance(Instance data) throws Exception {
        return 0.0;
    }
    
    public String toString(){
        String output = "";
        return output;
    }

    public double[] distributionForInstance(Instance instance){
        double[] result = new double[instance.numAttributes()];
        if(predict(instance) == 1){
            result[0] = 1;
            result[1] = 0;
        }else{
            result[0] = 0;
            result[1] = 1;
        }
        return result;
    }

    // Activation Function
    private int sign(double n){
        if(n >= 0)
            return 1;
        else
            return -1;
    }
}