package weka.filters;

import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.*;
import weka.filters.SimpleBatchFilter;
import java.util.ArrayList;

public class HDA
  extends SimpleBatchFilter {

  private static final long serialVersionUID = 1L;
  private final boolean DEBUG = true;

  public String globalInfo() {
    return   "A simple batch filter that adds an additional attribute 'bla' at the end "
      + "containing the index of the processed instance.";
  }

  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.enableAllAttributes();
    result.enableAllClasses();
    result.enable(Capability.NO_CLASS);  //// filter doesn't need class to be set//
    return result;
  }

  protected Instances determineOutputFormat(Instances inputFormat) {
    Instances result = new Instances(inputFormat, 0);
    result.insertAttributeAt(new Attribute("bla"), result.numAttributes());
    return result;
  }
  
  /**
   * @param inst  a list of Instances , which will be seperated based on class.
   * @return      Returns a three layer ArrayList of type Double,
   *              the first level is each dataset seperated by class,
   *              second level is each Instance in the dataset,
   *              And the final level is the value of the attributes.
   */
  protected ArrayList<ArrayList<ArrayList<Double>>> 
    seperateDatasetByClass(Instances inst) {
    ArrayList<ArrayList<ArrayList<Double>>> disjointDataset 
      = new ArrayList<ArrayList<ArrayList<Double>>>(inst.numClasses());
    
    for (int i = 0; i < inst.numClasses(); ++i) {
      disjointDataset.add(new ArrayList<ArrayList<Double>>());
    }
    /*
     * We find the class of each instance, add a new "Instance" ie arraylist to 
     * that disjoint dataset, and then add each value of the attribute to it.
     */
    for (int i = 0; i < inst.numInstances(); i++) {
      final int class_value = (int)inst.instance(i).classValue();

      disjointDataset.get(class_value).add(
              new ArrayList<Double>(inst.numAttributes()));

      for (int n = 0; n < inst.numAttributes(); n++) {
        int pos = disjointDataset.get(class_value).size();
        if (n != inst.classIndex()) {
          disjointDataset.get(class_value).get(pos-1).add(
                  inst.instance(i).value(n));
        }
      }
    }
    return disjointDataset;
  }

  protected Instances process(Instances inst) {

    // double_matrix will be used to construct a matrix of the dataset.
    double double_matrix[][] = new double[inst.size()][inst.numAttributes()];
    // Construct all D_{i}
    ArrayList<ArrayList<ArrayList<Double>>> disjointDataset 
            = seperateDatasetByClass(inst);
    // Instances is just a ArrayList<Instance> 
    Instances result = new Instances(determineOutputFormat(inst), 0);

    if (DEBUG) {
      System.out.println("Instances as passed in\n" + inst);
      System.out.println("result at the start:\n" + result.numAttributes());
      System.out.println(result);
      System.out.println("instance class index is " + 
              inst.instance(0).classIndex() + " vs num attributes " +
              inst.numAttributes());
    }
    
    /* 
     * Copies each of the old instance values to the new instance, 
     * and then sets the attribute "bla" to the correct instance index.
     */
    for (int i = 0; i < inst.numInstances(); i++) {
      double[] values = new double[result.numAttributes()];
      for (int n = 0; n < inst.numAttributes(); n++) {
        values[n] = inst.instance(i).value(n);
      }
      // sets attribute bla
      values[values.length - 1] = i;
      // Copies over this instance of data to the array to construct the matrix.
      double_matrix[i] = values;
      // Adds the instance, which is just an array for weka.
      result.add(new DenseInstance(1, values));
    }

    // Matrix with each row being a data point, last column is the class it belongs to.
    Matrix matrix = new Matrix(double_matrix);

    if (DEBUG) {
      System.out.println("Matrix was constructed as:");
      System.out.println("first columns are attributes values, and second last column is class number, last column is Attribute bla:\n" + matrix);
      System.out.println("We have constructed the following disjointDataset");
      for (int i = 0; i < disjointDataset.size(); ++i) {
          System.out.println("\ndisjointDataset number " + i + " is \n");
        for (int j = 0; j < disjointDataset.get(i).size(); ++j) {
          for (int k = 0; k < disjointDataset.get(i).get(j).size(); ++k) {
            System.out.print(" " + disjointDataset.get(i).get(j).get(k));
          }
          System.out.println("");
        }
      }
    }
    return result;
  }

  public static void main(String[] args) {
    runFilter(new HDA(), args);
  }
}
