import java.io.File;
import java.text.DecimalFormat;

import weka.classifiers.Classifier;
import weka.core.*;

/** University of Central Florida
 *CAP4630 Artificial Intelligence - Fall 2017
 *Perceptron Classifier by Julian Quitian & Ley Nezifort
 */

public class LogisticPerceptron implements weka.classifiers.Classifier{
    // Input file name
    private String fileName;

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

    // Number of times Perceptron is updated
    private int updateCount = 0;

    public LogisticPerceptron(String[] options) throws Exception{
        // Print the header for the output
		System.out.println("\nUniversity of Central Florida ");
		System.out.println("CAP4630 Artificial Intelligence - Fall 2017");
        System.out.println("Perceptron Classifier by Julian Quitian & Ley Nezifort \n");
        
        // Process Arguments
        this.fileName = options[0];
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

        // Number of data samples
        int dataSize = data.numInstances();

        // Get number of attributes by looking at first instance in dataset
        int numAttributes = data.firstInstance().numAttributes();
          
        // Make sure there are instances in the set. Return if otherwise.
		if(dataSize == 0){
			return;
        }
        
        // Make sure there are no more than two output classes available
        if(data.numClasses() > 2){
            return;
        }
        
        // Set weight vector size. Note: All initialized to 0.0, as required
        weights = new double[numAttributes];

        // Initializes the weights of all of the attributes in the first instance to 0.0
		for(int i = 0; i < data.firstInstance().numAttributes(); i++)
		{
            data.firstInstance().attribute(i).setWeight(0.0);
            weights[i] = 0.0;
        }

        // Run as many epochs as indicated by user
        for(int epochCounter = 0; epochCounter < this.numEpochs; epochCounter++){

            System.out.print("Epoch " + epochCounter + ": ");
            
            // Go through each sample in current data sample
            for(int sampleCounter = 0; sampleCounter < dataSize; sampleCounter++){
                // if(epochCounter == 2){
                //     if(sampleCounter == 0){
                //         System.out.println();
                //         System.out.print("Weights: ");
                //         for(int i = 0; i < weights.length; i++){
                //             System.out.print(weights[i] + " ");
                //         }
                //         System.out.println();
                //         System.exit(1);
                //     }
                // }

                // Current instance
                Instance currentInstance = data.instance(sampleCounter);

                // Perform summation of all attribute * weight values
                double sum = 0.0;
                for(int currentAtr = 0; currentAtr < numAttributes - 1; currentAtr++){
                    // sum += currentInstance.attribute(currentAtr).weight() * currentInstance.value(currentAtr);
                    sum += weights[currentAtr] * currentInstance.value(currentAtr);
                }
                
                // Account for bias weight
                sum += weights[weights.length - 1];

                // if(epochCounter == 2){
                //     if(sampleCounter == 0){
                //         for(int i = 0; i < weights.length; i++){
                //             System.out.println();
                //             System.out.print(weights[i] + " ");
                //         }
                //         System.out.println();
                //         System.out.println(sum);
                //         System.exit(1);
                //     }
                // }

                // This is the result of the Perceptron. If output does not match expected output, weights will be updated.   
                //int output = predict(currentInstance);             
                double temp = logisticFunction(sum);
                // if(epochCounter == 2){
                //     if(sampleCounter == 0){
                //         System.out.println();
                //         System.out.println(temp);
                //         System.exit(1);
                //     }
                // }

                // Scale
                temp = 2 * (temp) - 1;
                
                // if(epochCounter == 2){
                //     if(sampleCounter == 0){
                //         System.out.println();
                //         System.out.println(temp);
                //         System.exit(1);
                //     }
                // }

                int output;
                if(temp >= 0){
                    output = 1;
                }else{
                    output = -1;
                }
                
                // Expected output is last value of each attribute; map +1 to distribution for class a, and -1 for class b.
                int expectedOutput = (int)((currentInstance.value(currentInstance.attribute(numAttributes - 1)) == 0) ? 1 : -1);

                // if(sampleCounter == 1){
                //     System.out.println("Expected: " + expectedOutput + "\t Actual: " + output);
                //     System.exit(1);
                // }

                if(output != expectedOutput){
                    System.out.print("0");
					
					// Increase the number of times the weight has been updated by 1
					this.updateCount++;

                    double exponential = Math.exp(-1.0 * this.squashingParameter * sum);
                    // Calculate derivative
                    double fPrimeNet = (this.squashingParameter * exponential) / Math.pow(1 + exponential, 2);

                    // Calculate error and scale from -1 to 1
                    double error = expectedOutput - output;
                    error = 2.0 * (error + 2.0);
                    error = (error / 4.0) - 1.0;

                    for(int m = 0; m < numAttributes; m++){

                        double deltaW = this.learningRate * error * fPrimeNet * currentInstance.value(m);

                        // if(sampleCounter == 2){
                        //     if(m == 1){
                        //         System.out.println();
                        //         System.out.println(this.learningRate);
                        //         System.out.println(expectedOutput);
                        //         System.out.println(output);
                        //         System.out.println(currentInstance.value(m));
                        //         System.out.println(fPrimeNet);
                        //         System.out.println(deltaW);
                        //         System.exit(1);
                        //     }
                        // }

                        data.attribute(m).setWeight(data.attribute(m).weight() + deltaW);
                        weights[m] = weights[m] + deltaW;

                        // if(sampleCounter == 1){
                        //     if(m == 1){
                        //         System.out.println();
                        //         System.out.println(data.attribute(m).weight());
                        //         System.out.println(weights[m]);
                        //         System.exit(1);
                        //     }
                        // }
                    }

                    weights[weights.length - 1] += this.learningRate * error * fPrimeNet;

                }else{
                    System.out.print("1");
                }
            }
            // Move on to next epoch
			System.out.println();
        }
        // Store all of the weights in data back in the weights[] array
		for(int i = 0; i < numAttributes - 1; i++){
			this.weights[i] = data.attribute(i).weight();
        }
    }

    // Predicts +1 or -1 value for the classification of a sample instance.
    // public int predict(Instance instance){
    public int predict(Instance instance){
        // Holds all attribute values
        double[] attributes = new double[instance.numAttributes() - 1];

        // Fill input vector with corresponding attribute values
        for(int i = 0; i < instance.numAttributes() - 1; i++){
            attributes[i] = instance.value(i);
        }

        double sum = 0;
        // Run initial prediction function, which excludes bias
        for(int i = 0; i < instance.numAttributes() - 1; i++){
            sum += attributes[i] * instance.attribute(i).weight();
        }

        // Account for bias by adding it's corresponding weight to the current sum
        sum += instance.attribute(instance.numAttributes() - 1).weight() * this.bias;

        // int output = thresholdFunction(sum);
        // return output;

        double fnet = 1.0 / (1 + Math.pow(Math.E, (-1.0) * this.squashingParameter * sum));
        return (int)fnet;
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
        // Compiles a string consisting of the values of the final weights
		String finalWeights = "";
		
		// Creates a new instance of Decimal Format to restrict the print-outs of the 
		// final weights as shown in simple-out.png
		DecimalFormat df = new DecimalFormat("#0.000");
		
		// Creates the string containing the final weights
		for(int i = 0; i < this.weights.length; i++)
		{
            finalWeights += (df.format(this.weights[i]) + "\n");
		}
	
		// Prints out the required data
		System.out.println("Source file: " + this.fileName);
		System.out.println("Training epochs: " + this.numEpochs);
		System.out.println("Learning rate: " + this.learningRate + "\n");
        
        System.out.println("Total # weight updates = " + this.updateCount);
		System.out.println("Final weights: \n" + finalWeights); 
        return " ";
    }

    @Override
    public double[] distributionForInstance(Instance instance){
        double[] result = new double[instance.numAttributes()];
        if(predict(instance) == 1){
            result[0] = 1.0;
            result[1] = 0.0;
        }else{
            result[0] = 0.0;
            result[1] = 1.0;
        }
        return result;
    }

    // Activation Function
    private int thresholdFunction(double n){
        if(n >= 0)
            return 1;
        else
            return -1;

        //return (int)(1.0 / (1.0 + Math.pow(Math.E, (-1) * this.squashingParameter * n)));
    }

    public double logisticFunction(double sum){
        double fnet = 1.0 / (1 + Math.pow(Math.E, (-1.0) * this.squashingParameter * sum));
        return fnet;
    }
}