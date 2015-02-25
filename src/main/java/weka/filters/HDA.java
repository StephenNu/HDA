package weka.filters;

import weka.core.*;
import weka.core.Capabilities.*;
import weka.filters.*;
import weka.core.matrix.*;

public class HDA
  extends SimpleBatchFilter {

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
    weka.core.matrix.Matrix matrix = new weka.core.matrix.Matrix(double_matrix);
    if (DEBUG) {
      System.out.println("Matrix was constructed as:");
      System.out.println("first columns are attributes values, and last column is class number:\n" + matrix);
    }
    return result;
  }

  public static void main(String[] args) {
    runFilter(new HDA(), args);
  }
}
