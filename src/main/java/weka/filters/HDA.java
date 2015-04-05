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
    return   "A simple batch filter that adds an additional attribute "
     +"'bla' at the end containing the index of the processed instance.";
  }

  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.enableAllAttributes();
    result.enableAllClasses(); 
    //// filter doesn't need class to be set//
    result.enable(Capability.NO_CLASS);
    return result;
  }

  protected Instances determineOutputFormat(Instances inputFormat) {
    Instances result = new Instances(inputFormat, 0);
    result.insertAttributeAt(new Attribute("bla"), result.numAttributes());
    return result;
  }
  
  /**
   * @param inst  a list of Instances , which will be seperated based on class.
   * @return      Returns the passed in instances seperated by class values.
   */
  protected ArrayList<Instances> seperateDatasetByClass(Instances inst) {
    int numAtt = inst.numAttributes() - 1;
    ArrayList<Instances> disjointDataset 
            = new ArrayList<Instances>(inst.numClasses());

    for (int i = 0; i < inst.numClasses(); ++i) {
      disjointDataset.add(new Instances(inst, 0));
    }
    /*
     * We find the class of each instance, add a new "Instance" to
     * that disjoint dataset, and then add each value of the attribute to it.
     */
    for (int i = 0; i < inst.numInstances(); i++) {
      final int class_value = (int)inst.instance(i).classValue();
      double[] values = new double[numAtt];
      for (int n = 0; n < inst.numAttributes(); n++) {
        if (n != inst.instance(i).classIndex()) {
          values[n] = inst.instance(i).value(n);
        }
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
  protected ArrayList<Matrix> findSampleMeans(
          ArrayList<Instances> datasets) {
    ArrayList<Matrix> medians = new ArrayList<Matrix>();
    for (int i = 0; i < datasets.size(); ++i) {
      if (datasets.get(i).numInstances() > 0) {
        double[] values 
                = new double[datasets.get(i).instance(0).numAttributes()];
        for (int k = 0; k < datasets.get(i).numInstances(); ++k) {
          Instance current_instance = datasets.get(i).instance(k);
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
        median_i.timesEquals(1.0/(double)datasets.get(i).size());
        medians.add(median_i);
      } else {
        System.out.println("No instances found for class value of " + i);
      }
    }
    return medians;
  }

  /**
   * @param datasets    A list of instances representing each dataset
   *                    seperated by classes.
   * @return            Returns a list of covariance matricies each one
   *                    related to the dataset seperated by class.
   */
  protected ArrayList<Matrix> findCovarianceMatrices(
          ArrayList<Instances> datasets) {
    ArrayList<Matrix> covarianceMatrices = new ArrayList<Matrix>();
    ArrayList<Matrix> sampleMeans = findSampleMeans(datasets);
    for (int i = 0; i < datasets.size(); ++i) {
      if (datasets.get(i).numInstances() > 0) {
        double[][] val
                 = new double[datasets.get(i).instance(0).numAttributes()]
                             [datasets.get(i).instance(0).numAttributes()];
        Matrix covariance_i = new Matrix(val);
        for (int k = 0; k < datasets.get(i).numInstances(); ++k) {
          Instance current_instance = datasets.get(i).instance(k);
          double[] values = new double[current_instance.numAttributes()];
          int l = 0;
          for (int j = 0; j < current_instance.numAttributes(); ++j) {
            if (current_instance.classIndex() != j) {
              values[l] += current_instance.value(j);
              ++l;
            }
          }
          Matrix single_example = new Matrix(values, values.length);
          single_example.minusEquals(sampleMeans.get(i));
          single_example = single_example.times(single_example.transpose());
          covariance_i.plusEquals(single_example);
        }
        covariance_i.timesEquals(1.0/(double)datasets.get(i).size());
        covarianceMatrices.add(covariance_i);
      } else {
        System.out.println("No instances found for class value of " + i);
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
  protected ArrayList<Double> calculateProbability(
          ArrayList<Instances> datasets) {
    ArrayList<Double> probabilities = new ArrayList<Double>();
    double sum = 0.0;
    for (Instances inst : datasets) {
      sum += inst.numInstances();
    }

    for (Instances inst : datasets) {
      probabilities.add(new Double(((double)inst.numInstances())/sum));
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
  protected ArrayList<ArrayList<Double>> calculateRelativeProbability(
          ArrayList<Double> probabilities) {
    ArrayList<ArrayList<Double>> relativeProbabilities 
            = new ArrayList<ArrayList<Double>>();
    for (int i = 0; i < probabilities.size(); ++i) {
      relativeProbabilities.add(new ArrayList<Double>());
      for (int j = 0; j < probabilities.size(); ++j) {
        double prob_i = probabilities.get(i).doubleValue();
        double sampleSize = prob_i + probabilities.get(j).doubleValue();
        relativeProbabilities.get(i).add(new Double(prob_i/sampleSize));
      }
    }
    return relativeProbabilities;
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
  protected ArrayList<ArrayList<Matrix>> betweenClassScatterMatricies(
      ArrayList<Matrix> sampleMeans) {
    ArrayList<ArrayList<Matrix>> ScatterMatricies 
            = new ArrayList<ArrayList<Matrix>>();
    for (int i = 0; i < sampleMeans.size(); ++i) {
      ScatterMatricies.add(new ArrayList<Matrix>());
      for (int j = 0; j < sampleMeans.size(); ++j) {
        Matrix scatter = sampleMeans.get(i).minus(sampleMeans.get(j));
        ScatterMatricies.get(i).add(scatter.times(scatter.transpose()));
      }
    }
    return ScatterMatricies;
  }

  protected Instances process(Instances inst) {

    // double_matrix will be used to construct a matrix of the dataset.
    double double_matrix[][] = new double[inst.size()][inst.numAttributes()];
    // Construct all D_{i}
    ArrayList<Instances> disjointDataset 
            = seperateDatasetByClass(inst);
    ArrayList<Matrix> sampleMeans = findSampleMeans(disjointDataset);
    ArrayList<Matrix> covarianceMatrices 
            = findCovarianceMatrices(disjointDataset);
    ArrayList<Double> probabilities = calculateProbability(disjointDataset);
    ArrayList<ArrayList<Matrix>> scatterMatricies 
            = betweenClassScatterMatricies(sampleMeans);
    ArrayList<ArrayList<Double>> relativeProbabilities 
            = calculateRelativeProbability(probabilities);
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

    // Matrix with each row being a data point, 
    // last column is the class it belongs to.
    Matrix matrix = new Matrix(double_matrix);

    if (DEBUG) {
      System.out.println("Matrix was constructed as:");
      System.out.println("first columns are attributes values, and second last"
          +" column is class number, last column is Attribute bla:\n" + matrix);
      System.out.println("We have constructed the following disjointDataset");
      for (int i = 0; i < disjointDataset.size(); ++i) {
          System.out.println("\ndisjointDataset number " + i + " is \n");
          for (int j = 0; j < disjointDataset.get(i).size(); ++j) {
            for (int k = 0; 
                k < disjointDataset.get(i).instance(j).numAttributes(); 
                ++k) {
              System.out.print(""+ disjointDataset.get(i).instance(j).value(k));
            }
            System.out.println("");
          }
      }
      for (int i = 0; i < sampleMeans.size(); ++i) {
        System.out.println("We found that sample mean " + i + " was\n");
        System.out.println(sampleMeans.get(i));
      }
      for (int i = 0; i < covarianceMatrices.size(); ++i) {
        System.out.println("We found that covariance matrix " + i + " was\n");
        System.out.println(covarianceMatrices.get(i));
      }
      for (int i = 0; i < probabilities.size(); ++i) {
        System.out.println("We found probability for class " + i + " was\n");
        System.out.println(probabilities.get(i));
      }
      for (int i = 0; i < scatterMatricies.size(); ++i) {
        for (int j = 0; j < scatterMatricies.get(i).size(); ++j) {
          System.out.println("We found scattermatrix [" + i + ", " + j + "]");
          System.out.println(scatterMatricies.get(i).get(j));
        }
      }
      for (int i = 0; i < relativeProbabilities.size(); ++i) {
        for (int j = 0; j < relativeProbabilities.get(i).size(); ++j) {
          System.out.println("We found relative-prob [" + i + ", " + j + "]");
          System.out.println(relativeProbabilities.get(i).get(j));
        }
      }
    }
    return result;
  }

    public static void main(String[] args) {
      runFilter(new HDA(), args);
    }
  }
