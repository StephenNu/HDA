package weka.filters;

import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.*;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.SimpleBatchFilter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

public class HDA
  extends SimpleBatchFilter implements OptionHandler {

  private class Pair implements Comparable<Pair> {
    public double eigenvalue;
    public double[] eigenvector;

    Pair(double val, double[] vect) {
      eigenvalue = val;
      eigenvector = vect;
    }

    public int compareTo(Pair other) {
      if (eigenvalue < other.eigenvalue) {
        return 1;
      } else if (eigenvalue > other.eigenvalue) {
        return -1;
      } else {
        return 0;
      }
    }
  }

  private static final long serialVersionUID = 1L;
  private final boolean DEBUG = false;

  private int dimension = 1;
  private double threshold = 1e-8;

  public void setDimension(int dim) {
    dimension = dim;
  }

  public void setThreshold(double thresh) {
    threshold = thresh;
  }

  public int getDimension() {
    return dimension;
  }

  public double getThreshold() {
    return threshold;
  }

  public String dimensionTipText() {
    return "Changes the resultant dimension of the data after HDA is applied.";
  }

  public String thresholdTipText() {
    return "Changes the lower limit for eigenvalues.";
  }

  /**
   * @return            Returns an enumeration describing the available options.
   */
  public Enumeration<Option> listOptions() {
    Vector<Option> ops = new Vector<Option>();

    for (Enumeration<Option> list_ops = super.listOptions(); 
         list_ops.hasMoreElements();) {
      ops.add(list_ops.nextElement());
    }
    ops.add(new Option("Specify the new dimension (default 1)", 
                       "dim", 1, "-dim <num>"));
    ops.add(new Option("Specify the new threshold (default 1e-8)",
                       "thresh", 1, "-thresh <num>"));

    return ops.elements();
  }

  /**
   * @param options                 The list of options to set.
   * @throws Exception              If an option is not supported.
   */
  public void setOptions(String[] options) throws Exception {
    super.setOptions(options);
    String tmpStr = Utils.getOption("dim", options);
    
    if (tmpStr.length() != 0) {
      setDimension(Integer.parseInt(tmpStr));
    } else {
      setDimension(1);
    }

    tmpStr = Utils.getOption("thresh", options);

    if (tmpStr.length() != 0) {
      setThreshold(Double.parseDouble(tmpStr));
    } else {
      setThreshold(1e-8);
    }
  }

  /**
   * @return          Returns the current setting of the filter.
   */
  public String[] getOptions() {
    String[] list_ops = super.getOptions();
    Vector<String> result = new Vector<String>();

    for (String o : list_ops) {
      result.add(o);
    }
    result.add("-dim");
    result.add("" + getDimension());
    result.add("-thresh");
    result.add("" + getThreshold());

    return result.toArray(new String[result.size()]);
  }


  public String globalInfo() {
    return "Performs Heteroscedastic Discriminant Analysis and transforms the " 
         + "data.\n"
         + "Dimensionality reduction is done by finding the eigenvectors "
         + "associated with the d (dimension) largest eigenvalues of a related "
         + "matrix and multiplying each instance by them.";
  }

  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();
    // Attributes
    result.enable(Capability.DATE_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.BINARY_ATTRIBUTES);
    // Class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.NUMERIC_CLASS);
    result.enable(Capability.BINARY_CLASS);
    return result;
  }
 
  protected Instances determineOutputFormat(Instances inputFormat) {
    ArrayList<Attribute> attinfo = new ArrayList<Attribute>();
    for (int i = 1; i <= dimension; ++i) {
      attinfo.add(new Attribute("Attribute " + i));
    }
    attinfo.add((Attribute)inputFormat.classAttribute().copy());
    Instances result = new Instances("Reduced data", attinfo, 0);
    result.setClassIndex(dimension);
    return result;
  }
 
  protected Instances process(Instances inst) throws Exception {
    if (dimension < 0) {
      throw new Exception("Cannot reduce dimension below 0");
    } else if (dimension >= inst.numAttributes()) {
      throw new Exception("Cannot reduce to a dimension greater " +
                          "than number of attributes");
    }
    if (threshold <= 0) {
      throw new Exception("Threshold must be > 0");
    }
    // double_matrix will be used to construct a matrix of the dataset.
    double double_matrix[][] = new double[inst.size()][inst.numAttributes()];
    // Construct all D_{i}
    HashMap<Integer, Instances> disjointDataset
            = separateDatasetByClass(inst);
    HashMap<Integer, Matrix> sampleMeans = findSampleMeans(disjointDataset);
    HashMap<Integer, Matrix> covarianceMatrices
            = findCovarianceMatrices(disjointDataset);
    HashMap<Integer, Double> probabilities
            = calculateProbability(disjointDataset);
    HashMap<Integer, HashMap<Integer, Matrix>> scatterMatrices
            = betweenClassScatterMatrices(sampleMeans);
    HashMap<Integer, HashMap<Integer, Double>> relativeProbabilities
            = calculateRelativeProbability(probabilities);
    Matrix withinClassScatter
            = withinClassScatterMatrix(covarianceMatrices, probabilities);
    HashMap<Integer, HashMap<Integer, Matrix>> combinedScatters
            = combineScatterMatrices(relativeProbabilities, covarianceMatrices);
    // Instances is just an ArrayList<Instance>
    Instances result = new Instances(determineOutputFormat(inst), 0);

    Matrix A = solution(withinClassScatter, scatterMatrices,
        relativeProbabilities, combinedScatters, covarianceMatrices, 
        probabilities);

    Instances reducedData = reduceDimension(inst, A);

    if (DEBUG) {
      System.out.println("Instances as passed in\n" + inst);
      System.out.println("result at the start:\n" + result.numAttributes());
      System.out.println(result);
      System.out.println("instance class index is " +
              inst.classIndex() + " vs num attributes " +
              inst.numAttributes());
    }

    if (DEBUG) {
      try {
        System.out.println("We have constructed the following disjointDataset");
        for (Map.Entry<Integer, Instances> entry : disjointDataset.entrySet()) {
            System.out.println("\ndisjointDataset number "
                + entry.getKey() + " is \n");
            for (int j = 0; j < entry.getValue().size(); ++j) {
              for (int k = 0;
                  k < entry.getValue().instance(j).numAttributes();
                  ++k) {
                System.out.print(""+ entry.getValue().instance(j).value(k));
              }
              System.out.println("");
            }
        }
        for (Map.Entry<Integer, Matrix> entry : sampleMeans.entrySet()) {
          System.out.println("We found that sample mean "
              + entry.getKey() + " was\n");
          System.out.println(sampleMeans.get(entry.getKey()));
        }
        for (Map.Entry<Integer, Matrix> entry : covarianceMatrices.entrySet()) {
          System.out.println("We found that covariance matrix "
              + entry.getKey() + " was\n");
          System.out.println(covarianceMatrices.get(entry.getKey()));
        }
        for (Map.Entry<Integer, Double> entry : probabilities.entrySet()) {
          System.out.println("We found probability for class "
             + entry.getKey() + " was\n");
          System.out.println(probabilities.get(entry.getKey()));
        }
        for (Map.Entry<Integer, HashMap<Integer, Matrix>> entry
                : scatterMatrices.entrySet()) {
          for (Map.Entry<Integer, Matrix> entry2
                  : scatterMatrices.get(entry.getKey()).entrySet()) {
            System.out.println("We found scattermatrix " +
                "[" + entry.getKey() + ", " + entry2.getKey()+ "]");
            System.out.println(
                scatterMatrices.get(
                  entry.getKey()
                ).get(
                  entry2.getKey()
                )
            );
          }
        }
        for (Map.Entry<Integer, HashMap<Integer, Double>> entry
            : relativeProbabilities.entrySet()) {
          for (Map.Entry<Integer, Double> entry2
              : relativeProbabilities.get(entry.getKey()).entrySet()) {
            System.out.println("We found relative-prob "
                +"[" + entry.getKey() + ", " + entry2.getKey()+ "]");
            System.out.println(
                relativeProbabilities.get(
                  entry.getKey()
                ).get(
                  entry2.getKey()
                )
            );
          }
        }
        System.out.println("We found the within class scatter to be");
        System.out.println(withinClassScatter);
        Map.Entry<Integer, Matrix> firstPair
              = covarianceMatrices.entrySet().iterator().next();
        System.out.println("We take the following matrix to 1/2 and then -1/2");
        System.out.println(firstPair.getValue());
        System.out.println("To 1/2");
        System.out.println(":" + matrixToOneHalf(
              firstPair.getValue(), true));
        System.out.println("To -1/2");
        System.out.println(":" + matrixToOneHalf(
              firstPair.getValue(), false));
        System.out.println("We take the following matrix and apply log");
        System.out.println(firstPair.getValue());
        System.out.println("log(A)");
        System.out.println(":\n" + matrixLog(firstPair.getValue()));

        System.out.println("Finding solution:");
        Matrix solution = solutionIteration(0, 1, withinClassScatter,
            scatterMatrices, relativeProbabilities, combinedScatters,
            covarianceMatrices, probabilities);
        System.out.println("SOLUTION IS THIS OKAY THANKS\n" + solution);

        System.out.println("Finding A:");
        Matrix summation = solution(withinClassScatter, scatterMatrices,
            relativeProbabilities, combinedScatters, covarianceMatrices,
            probabilities);
        System.out.println("A =\n" + summation);
        System.out.println("Reduced data:\n" + reducedData);

        //Put tests above this
        System.out.println("This line is to let you know everything finished");
      } catch (OutOfMemoryError E) {
        System.out.println("Debug strings were to large to be printed");
      }
    }
    return reducedData;
  }




  /**
   * @param inst  a list of Instances , which will be seperated based on class.
   * @return      Returns the passed in instances seperated by class values.
   */
  protected HashMap<Integer, Instances> separateDatasetByClass(
          Instances inst) {
    int numAtt = inst.numAttributes();
    HashMap<Integer, Instances> disjointDataset
            = new HashMap<Integer, Instances>();

    for (int i = 0; i < inst.numInstances(); ++i) {
      if (!disjointDataset.containsKey(inst.instance(i).classValue())) {
        disjointDataset.put(
            new Integer(
              (int)inst.instance(i).classValue()), new Instances(inst, 0));
      }
    }
    /*
     * We find the class of each instance, add a new "Instance" to
     * that disjoint dataset, and then add each value of the attribute to it.
     */
    for (int i = 0; i < inst.numInstances(); i++) {
      final int class_value = (int)inst.instance(i).classValue();
      double[] values = new double[numAtt];
      for (int n = 0; n < inst.numAttributes(); n++) {
        values[n] = inst.instance(i).value(n);
      }
      // Adds the instance, which is just an array for weka.
      disjointDataset.get(class_value).add(new DenseInstance(1, values));
    }
    return disjointDataset;
  }

  /**
   * @param datasets    A list of instances representing each dataset 
   *                    seperated by classes.
   * @return            Returns a list of single column matricies, i.e. a vector
   *                    each representing a mean of all sample data seperated by
   *                    classes.
   */
  protected HashMap<Integer, Matrix> findSampleMeans(
          HashMap<Integer, Instances> datasets) {
    HashMap<Integer, Matrix> medians = new HashMap<Integer, Matrix>();
    for (Map.Entry<Integer, Instances> entry : datasets.entrySet()) {
        Instances inst = entry.getValue();
        double[] values 
                = new double[inst.numAttributes()-1];
        for (int k = 0; k < inst.numInstances(); ++k) {
          Instance current_instance = inst.instance(k);
          int l = 0;
          for (int j = 0; j < current_instance.numAttributes(); ++j) {
            if (current_instance.classIndex() != j) {
              values[l] += current_instance.value(j);
              ++l;
            }
          }
        }
        // We construct a vector and divide it by the size of the datasets.
        Matrix median_i = new Matrix(values, values.length);
        if (inst.size() > 0) {
          median_i.timesEquals(1.0/(double)inst.size());
        }
        medians.put(entry.getKey(), median_i);
    }
    return medians;
  }

  /**
   * @param datasets    A list of instances representing each dataset
   *                    seperated by classes.
   * @return            Returns a list of covariance matricies each one
   *                    related to the dataset seperated by class.
   */
  protected HashMap<Integer, Matrix> findCovarianceMatrices(
          HashMap<Integer, Instances> datasets) {
    HashMap<Integer, Matrix> covarianceMatrices
            = new HashMap<Integer, Matrix>();
    HashMap<Integer, Matrix> sampleMeans = findSampleMeans(datasets);
    for (Map.Entry<Integer, Instances> entry : datasets.entrySet()) {
        Instances inst = entry.getValue();
        double[][] val
                 = new double[inst.numAttributes()-1]
                             [inst.numAttributes()-1];
        Matrix covariance_i = new Matrix(val);
        for (int k = 0; k < inst.numInstances(); ++k) {
          Instance current_instance = inst.instance(k);
          double[] values = new double[current_instance.numAttributes()-1];
          int l = 0;
          for (int j = 0; j < current_instance.numAttributes(); ++j) {
            if (current_instance.classIndex() != j) {
              values[l] += current_instance.value(j);
              ++l;
            }
          }
          Matrix single_example = new Matrix(values, values.length);
          single_example.minusEquals(sampleMeans.get(entry.getKey()));
          single_example = single_example.times(single_example.transpose());
          covariance_i.plusEquals(single_example);
        }
      if (inst.size() > 0) {
        covariance_i.timesEquals(1.0/(double)inst.size());
        covarianceMatrices.put(entry.getKey(), covariance_i);
      } else {
        System.out.println("In covariance no instances of class "
                           + entry.getKey());
      }
    }
    return covarianceMatrices;
  }

  /**
   * @param datasets    A list of instances representing each dataset
   *                    separated by clases.
   * @return            Returns a list of probabilities of each class
   *                    occuring in the provided dataset.
   */
  protected HashMap<Integer, Double> calculateProbability(
          HashMap<Integer, Instances> datasets) {
    HashMap<Integer, Double> probabilities = new HashMap<Integer, Double>();
    double sum = 0.0;
    for (Instances inst : datasets.values()) {
      sum += inst.numInstances();
    }

    for (Map.Entry<Integer, Instances> entry : datasets.entrySet()) {
      Instances inst = entry.getValue();
      probabilities.put(
              entry.getKey(), new Double(((double)inst.numInstances())/sum));
    }
    return probabilities;
  }

  /**
   * @param probabilities A list of probabilties for a given example to
   *                      belong to a specified class.
   *
   * @return              Returns a list of lists of probabilities, where
   *                      The value at [i,j] is the probability that given
   *                      an example, it will belong to class i when only
   *                      considering examples belonging to either class i or j
   */
  protected HashMap<Integer, HashMap<Integer, Double>>
        calculateRelativeProbability(HashMap<Integer, Double> probabilities) {
    HashMap<Integer, HashMap<Integer, Double>> relativeProbabilities
            = new HashMap<Integer, HashMap<Integer, Double>>();
    for (Integer idxi : probabilities.keySet()) {
      relativeProbabilities.put(idxi, new HashMap<Integer, Double>());
      for (Integer idxj : probabilities.keySet()) {
        double prob_i = probabilities.get(idxi).doubleValue();
        double sampleSize = prob_i + probabilities.get(idxj).doubleValue();
        relativeProbabilities.get(idxi).put
          (idxj, new Double(prob_i/sampleSize)
        );
      }
    }
    return relativeProbabilities;
  }

  /**
   * @param relativeProbabilities   A list of lists of probabilites that will 
   *                                be multiplied by the the corresponding 
   *                                covariance matrices and used to create the
   *                                combined scatter matrices.
   * @param covarianceMatrices      A list of covariance matrices for each 
   *                                class, to be multiplied by the relevant 
   *                                probabilities.
   *
   * @return                        Returns a list of a list of matrices, 
   *                                where the matrix at [i,j] is the combined 
   *                                scatter matrices i and j
   */
  protected HashMap<Integer, HashMap<Integer, Matrix>> combineScatterMatrices(
      HashMap<Integer, HashMap<Integer, Double>> relativeProbabilities, 
      HashMap<Integer, Matrix> covarianceMatrices) {
    HashMap<Integer, HashMap<Integer, Matrix>> combinedScatters 
           = new HashMap<Integer, HashMap<Integer, Matrix>>(); 
    for (Integer i : covarianceMatrices.keySet()) {
      combinedScatters.put(i, new HashMap<Integer, Matrix>());
      Matrix covar_i = covarianceMatrices.get(i);
      for (Integer j : covarianceMatrices.keySet()) {
        Matrix covar_j = covarianceMatrices.get(j);
        Double prob_i = relativeProbabilities.get(i).get(j);
        Double prob_j = relativeProbabilities.get(j).get(i);
        Matrix scatter_ij = covar_i.times(prob_i).plus(covar_j.times(prob_j));

        combinedScatters.get(i).put(j, scatter_ij);
        }
      }

    return combinedScatters;
  }

  /**
   * @param sampleMeans A list of vectors that represent the ith mean
   *                    for each of the i classes.
   *
   * @return            Returns a list of list of matricies, where the
   *                    matrix at [i,j] is the Scatter matrix formed by
   *                    (meani - meanj)*transpose(meani - meanj)
   *
   */
  protected HashMap<Integer, HashMap<Integer, Matrix>>
    betweenClassScatterMatrices(HashMap<Integer, Matrix> sampleMeans) {
    HashMap<Integer, HashMap<Integer, Matrix>> ScatterMatrices
            = new HashMap<Integer, HashMap<Integer, Matrix>>();
    for (Integer idxi : sampleMeans.keySet()) {
      ScatterMatrices.put(idxi, new HashMap<Integer, Matrix>());
      for (Integer idxj : sampleMeans.keySet()) {
        Matrix scatter = sampleMeans.get(idxi).minus(sampleMeans.get(idxj));
        ScatterMatrices.get(idxi).put
          (idxj, scatter.times(scatter.transpose())
        );
      }
    }
    return ScatterMatrices;
  }

  /**
   * @param covarianceMatrices  A list of covariance matricies for each class,
   *                            to be combined with the probabilities to form
   *                            the Within Class Scatter Matrix.
   * @param probabilities       A list of probabilities for each class i.
   *
   * @return                    Returns all the sum of all covariance matricies
   *                            times their respective probability.
   */
  protected Matrix withinClassScatterMatrix(
          HashMap<Integer, Matrix> covarianceMatrices,
          HashMap<Integer, Double> probabilities) {
    if (covarianceMatrices.size() == 0 &&
        covarianceMatrices.size() == probabilities.size()) {
      System.out.println("Sorry no covarianceMatrices.");
      System.out.println("Or probabilities were passed in.");
      return null;
    } else {
      Matrix withinClassScatter = null;
      for (Integer idxi : covarianceMatrices.keySet()) {
        if (withinClassScatter == null) {
          withinClassScatter = covarianceMatrices.get(idxi).copy().times(probabilities.get(idxi));
        } else {
          withinClassScatter.plusEquals(
                  covarianceMatrices.get(idxi).times(probabilities.get(idxi))
        );
        }
      }
      return withinClassScatter;
    }
  }

  /**
   * @param A                   A matrix that will be raised to 1/2 or -1/2
   * @param positivePower       If true the matrix will be raised to 1/2
   *                            Otherwise the matrix will be raised to -1/2
   *
   * @return                    Returns the matrix A^(1/2) if positivePower is
   *                            true, and A^(-1/2) otherwise.
   */
  protected Matrix matrixToOneHalf(Matrix A, boolean positivePower) 
      throws Exception {
    EigenvalueDecomposition values = new EigenvalueDecomposition(A);
    Matrix M = values.getD();
    int M_rows = M.getRowDimension();
    int M_cols = M.getColumnDimension();

    for (int i = 0; i < M_rows && i < M_cols; ++i) {
      if (M.getArray()[i][i] < threshold) {
        M.getArray()[i][i] = threshold;
      }
    }
    
    for (int i = 0; i < M_rows && i < M_cols; ++i) {
      M.getArray()[i][i]
          = (positivePower) ?
              Math.sqrt(M.getArray()[i][i]) : 1.0/Math.sqrt(M.getArray()[i][i]);
    }
    if (DEBUG) {
      System.out.println("We have a positivepower is " + positivePower);
      System.out.println("Working with eigenvectors \n" + values.getV());
      System.out.println("and eignvalues \n" + values.getD());
      System.out.println("and changed values\n" + M);
    }
    Matrix AOneHalf = values.getV().copy();
    AOneHalf = AOneHalf.times(M);
    AOneHalf = AOneHalf.times(values.getV().inverse());
    return AOneHalf;
  }

  /**
   * @param A                   A matrix to apply the log too.
   *
   * @return                    Returns the matrix log(A)
   */
  protected Matrix matrixLog(Matrix A) throws Exception {
    EigenvalueDecomposition values = new EigenvalueDecomposition(A);
    Matrix M = values.getD();
    int M_rows = M.getRowDimension();
    int M_cols = M.getColumnDimension();
    for (int i = 0; i < M_rows && i < M_cols; ++i) {
      if (M.getArray()[i][i] < threshold) {
        M.getArray()[i][i] = threshold;
      }
    }
    for (int i = 0; i < M_rows && i < M_cols; ++i) {
      M.getArray()[i][i] = Math.log(M.getArray()[i][i]);
    }
    if (DEBUG) {
      System.out.println("Working with eigenvectors \n" + values.getV());
      System.out.println("and eignvalues \n" + values.getD());
      System.out.println("and changed values\n" + M);
    }
    Matrix logA = values.getV().copy();
    logA = logA.times(M);
    logA = logA.times(values.getV().inverse());
    return logA;
  }

  /**
   * @param inst                A list of instances, which will be added to
   *                            a matrix and transformed.
   * @param A                   A matrix holding the eigenvectors associated
   *                            with the largest eigenvalues.
   * @return                    The transformed (reduced) instances.
   */
  protected Instances reduceDimension(Instances inst, Matrix A) {
    int num_att = inst.numAttributes();
    int num_inst = inst.numInstances();

    double[][] data_array = new double[num_att-1][num_inst];

    for (int i = 0; i < num_inst; ++i) {
      int l = 0;
      Instance curr_inst = inst.instance(i);
      for (int j = 0; j < num_att; ++j) {
        if (j != curr_inst.classIndex()) {
          data_array[l][i] = curr_inst.value(j);
          ++l;
        }
      }
    }

    Matrix X = new Matrix(data_array);
    Matrix Y = A.times(X);

    Instances reduced_inst = determineOutputFormat(inst);

    for (int i = 0; i < num_inst; ++i) {
      DenseInstance new_inst = new DenseInstance(dimension+1);
      for (int j = 0; j < dimension; ++j) {
        new_inst.setValue(j, Y.get(j,i));
      }
      new_inst.setValue(dimension, inst.instance(i).classValue());
      reduced_inst.add(new_inst);
    }
    return reduced_inst;
  }

  /**
   * @param withinClassScatter      A matrix holding the sum of all covariance
   *                                matrices multiplied by their respective
   *                                probabilities.
   * @param betweenClassScatters    A list of matrices holding the scatter
   *                                matrix for each pair of classes.
   * @param relativeProbabilities   A list of all the relative probabilities
   *                                for each pair of classes.
   * @param combinedScatters        A list of combined scatter matrices for
   *                                each pair of classes.
   * @param covariances             The covariances matrices for each class.
   * @param probabilities           The probabilities for each class.
   * @return                        Returns a matrix of the eigenvectors
   *                                associated with the d (dimension) largest
   *                                eigenvalues.
   */
  protected Matrix solution(Matrix withinClassScatter,
      HashMap<Integer, HashMap<Integer, Matrix>> betweenClassScatters,
      HashMap<Integer, HashMap<Integer, Double>> relativeProbabilities,
      HashMap<Integer, HashMap<Integer, Matrix>> combinedScatters,
      HashMap<Integer, Matrix> covariances,
      HashMap<Integer, Double> probabilities) throws Exception {
    int rows = withinClassScatter.getRowDimension();
    int columns = withinClassScatter.getColumnDimension();
    Matrix summation = new Matrix(rows, columns);

    for (Integer i : covariances.keySet()) {
      for (Integer j : covariances.keySet()) {
        if (i < j) {
          summation = summation.plus(solutionIteration(i, j, withinClassScatter,
                betweenClassScatters, relativeProbabilities, combinedScatters,
                covariances, probabilities));
        }
      }
    }
    EigenvalueDecomposition values = new EigenvalueDecomposition(summation);

    ArrayList<Pair> eigenpairs = new ArrayList<Pair>();
    double[][] eigenvectors = values.getV().transpose().getArray();
    for (int i = 0; i < columns; ++i) {
      eigenpairs.add(new Pair(values.getD().getArray()[i][i], eigenvectors[i]));
    }
    
    Collections.sort(eigenpairs);

    double[][] solution = new double[dimension][rows];

    for (int i = 0; i < dimension; ++i) {
      solution[i] = eigenpairs.get(i).eigenvector;
    }

    Matrix max_eigens = new Matrix(solution);

    if (DEBUG) {
      System.out.println("Eigenvectors \n" + values.getV());
      System.out.println("Eigenvalues \n" + values.getD());
      System.out.println("Eigenvalues Max \n" + max_eigens);
    }

    return max_eigens;
  }

  /**
   * @param i                       The index of the first class used
   * @param j                       The index of the second class used
   * @param withinClassScatter      A matrix holding the sum of all covariance
   *                                matrices multiplied by their respective
   *                                probabilities.
   * @param betweenClassScatters    A list of matrices holding the scatter
   *                                matrix for each pair of classes.
   * @param relativeProbabilities   A list of all the relative probabilities
   *                                for each pair of classes.
   * @param combinedScatters        A list of combined scatter matrices for
   *                                each pair of classes.
   * @param covariances             The covariances matrices for each class.
   * @param probabilities           The probabilities for each class.
   * @return                        A matrix holding the results of a single
   *                                iteration of the HDA algorithm.
   */
  protected Matrix solutionIteration(int i, int j, Matrix withinClassScatter,
      HashMap<Integer, HashMap<Integer, Matrix>> betweenClassScatters,
      HashMap<Integer, HashMap<Integer, Double>> relativeProbabilities,
      HashMap<Integer, HashMap<Integer, Matrix>> combinedScatters,
      HashMap<Integer, Matrix> covariances,
      HashMap<Integer, Double> probabilities) throws Exception {
    double prob = probabilities.get(i) * probabilities.get(j);
    double rel_prob_i = relativeProbabilities.get(i).get(j);
    double rel_prob_j = relativeProbabilities.get(j).get(i);
    double rel_prob_inverse = 1 / (rel_prob_i * rel_prob_j);

    Matrix pos_root_within = matrixToOneHalf(withinClassScatter, true);
    Matrix neg_root_within = matrixToOneHalf(withinClassScatter, false);
    Matrix within_inverse = withinClassScatter.inverse();

    Matrix combined_scatter = combinedScatters.get(i).get(j);
    Matrix between_scatter = betweenClassScatters.get(i).get(j);
    Matrix covariance_i = covariances.get(i);
    Matrix covariance_j = covariances.get(j);

    //A = Pi*Pj * Sw^(-1) * Sw^(1/2)
    Matrix solution = within_inverse;
    solution = solution.times(prob);
    solution = solution.times(pos_root_within);

    //B = Sw^(-1/2) * Sij * Sw(-1/2)
    Matrix within_combined = neg_root_within;
    within_combined = within_combined.times(combined_scatter);
    within_combined = within_combined.times(neg_root_within);

    //J = PIj * log(Sw^(-1/2) * Sj * Sw^(-1/2))
    Matrix log_within_j = neg_root_within;
    log_within_j = log_within_j.times(covariance_j);
    log_within_j = log_within_j.times(neg_root_within);
    log_within_j = matrixLog(log_within_j);
    log_within_j = log_within_j.times(rel_prob_j);

    //I = PIi * log(Sw^(-1/2) * Si * Sw^(-1/2))
    Matrix log_within_i = neg_root_within;
    log_within_i = log_within_i.times(covariance_i);
    log_within_i = log_within_i.times(neg_root_within);
    log_within_i = matrixLog(log_within_i);
    log_within_i = log_within_i.times(rel_prob_i);

    //L = 1/(PIi * PIj) * log(B) - J - I
    Matrix log_part = matrixLog(within_combined);
    log_part = log_part.minus(log_within_i);
    log_part = log_part.minus(log_within_j);
    log_part = log_part.times(rel_prob_inverse);

    //F = B^(-1/2) * Sw^(-1/2) * SEij * Sw^(-1/2) * B^(-1/2) + L
    Matrix root_within_combined = matrixToOneHalf(within_combined, false);
    Matrix bracket_part = root_within_combined;
    bracket_part = bracket_part.times(neg_root_within);
    bracket_part = bracket_part.times(between_scatter);
    bracket_part = bracket_part.times(neg_root_within);
    bracket_part = bracket_part.times(root_within_combined);
    bracket_part = bracket_part.plus(log_part);

    //Solution = A * F * Sw^(1/2)
    solution = solution.times(bracket_part);
    solution = solution.times(pos_root_within);

    return solution;
  }

    public static void main(String[] args) {
      runFilter(new HDA(), args);
    }
  }
