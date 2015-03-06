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

  protected Instances process(Instances inst) {
    double double_matrix[][] = new double[inst.size()][inst.numAttributes()];
    ArrayList<Instances> disjointDataset = new ArrayList<Instances>();
    Instances result = new Instances(determineOutputFormat(inst), 0);
    for (int i = 0; i < inst.numInstances(); i++) {
      double[] values = new double[result.numAttributes()];
      for (int n = 0; n < inst.numAttributes(); n++) {
        values[n] = inst.instance(i).value(n);
      }
      values[values.length - 1] = inst.instance(i).classValue();
      double_matrix[i] = values;
      result.add(new DenseInstance(1, values));
    }
    // Matrix with each row being a data point, last column is the class it belongs to.
    Matrix matrix = new Matrix(double_matrix);
    if (DEBUG) {
      System.out.println("Matrix was constructed as:");
      System.out.println("first columns are attributes values, and last column is class number:\n" + matrix);
      System.out.println("We have constructed the following disjointDataset");
      for (int i = 0; i < disjointDataset.size(); ++i) {
        System.out.println("disjointDataset number " + i + " is \n" + disjointDataset.get(i));
      }
    }
    return result;
  }

  public static void main(String[] args) {
    runFilter(new HDA(), args);
  }
}
