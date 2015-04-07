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
import java.util.HashMap;

import weka.core.Instances;
import weka.core.TestInstances;
import weka.filters.AbstractFilterTest;
import weka.filters.Filter;
import weka.filters.HDA;

import junit.framework.Test;
import junit.framework.TestSuite;

public class HDATest
  extends AbstractFilterTest {

  private static final double DELTA = 1e-15;

  public HDATest(String name) {
    super(name);
  }

  protected void setUp() throws Exception {
    super.setUp();

    m_Instances.deleteAttributeAt(6);  // Attribute Numeric type missing values
    //m_Instances.deleteAttributeAt(5);// Attribute Date type missing values
    m_Instances.deleteAttributeAt(4);  // Attribute Nominal type missing values
    m_Instances.deleteAttributeAt(3);  // Attribute String type missing values
    //m_Instances.deleteAttributeAt(2);// Attribute Numeric Type
    m_Instances.deleteAttributeAt(1);  // Attribute Nominal type
    m_Instances.deleteAttributeAt(0);  // Attribute String type

    // Half to class one, half to class two.
    for (int i = 0; i < m_Instances.numInstances(); ++i) {
      if (i < m_Instances.numInstances()/2) {
        m_Instances.instance(i).setValue(1, 0);
      } else {
        m_Instances.instance(i).setValue(1, 1);
      }
    }
    m_Instances.setClassIndex(1);
  }

  /** 
   * @return Creates a default HDA filter.
   */
  public Filter getFilter() {
    return new HDA();
  }

  /**
   * Gets the data after being filtered.
   *
   * @return              The dataset has been reduced in dimensions.
   * @throws Exception    An exception is thrown when the data generation fails
  protected Instances getFilterClassiferData() throws Exception {
    System.out.println("Class index " + m_Instances.instance(0).classIndex());
    m_Instances.setClassIndex(m_Instances.numAttributes()-1);
    return m_Instances;
  }

   */

  // Place tests here of the form
  // public void test<test-name>() { ... test code ... }


  public void testSeparateDatasetByClass() {
    final int NUM_CLASSES = 4;
    final int NUM_INSTANCES = 5;

    ArrayList<Attribute> attinfo = new ArrayList<Attribute>();
    attinfo.add(new Attribute("example-ID"));
    attinfo.add(new Attribute("class-ID"));
    Instances testInst = new Instances("Test instances", attinfo, 0);
    testInst.setClassIndex(1);
    int id = 0;
    // Set up NUM_CLASSES with NUM_INSTANCES each with unique ID
    for (int i = 0; i < NUM_CLASSES; ++i) {
      for (int j = 0; j < NUM_INSTANCES; ++j ) {
        Instance inst = new DenseInstance(2);
        inst.setValue(0, id);
        inst.setValue(1, i);
        ++id;
        testInst.add(inst);
      }
    }

    // Process the instances.
    HDA filter = (HDA)getFilter();
    HashMap<Integer, Instances> disjointDataset
            = filter.separateDatasetByClass(testInst);
    HashMap<Integer, ArrayList<Integer>> actual_separation
            = new HashMap<Integer, ArrayList<Integer>>();

    // Check actual output equals expected output.
    for (int i = 0; i < NUM_CLASSES; ++i) {
      assertTrue(disjointDataset.containsKey(i));
      actual_separation.put(i, new ArrayList<Integer>());
      for (int j = 0; j < NUM_INSTANCES; ++j) {
        actual_separation.get(i).add(
                (int)disjointDataset.get(i).instance(j).value(0)
        );
      }
    }
    int expected_id = 0;
    for (int i = 0; i < NUM_CLASSES; ++i) {
      ArrayList<Integer> list = actual_separation.get(i);
      for (int j = 0; j < NUM_INSTANCES; ++j) {
        assertTrue(list.contains(expected_id));
        ++expected_id;
      }
    }
  }

  public void testCombineScatterMatrices() {
    final double PROB_00 = 0.75;
    final double PROB_01 = 3;
    final double PROB_10 = 1.75;
    final double PROB_11 = 2.33;
    final Matrix COVARIANCE_0 = new Matrix(3, 3, 0.8);
    final Matrix COVARIANCE_1 = new Matrix(3, 3, 2.73);

    double ANS_00 = 1.2;
    double ANS_01 = 7.1775;
    double ANS_10 = 7.1775;
    double ANS_11 = 12.7218;

    HashMap<Integer, HashMap<Integer, Double>> testProbabilities
        = new HashMap<Integer, HashMap<Integer, Double>>();

    testProbabilities.put(0, new HashMap<Integer, Double>());
    testProbabilities.put(1, new HashMap<Integer, Double>());
    testProbabilities.get(0).put(0, PROB_00);
    testProbabilities.get(0).put(1, PROB_01);
    testProbabilities.get(1).put(0, PROB_10);
    testProbabilities.get(1).put(1, PROB_11);

    HashMap<Integer, Matrix> covarianceMatrices 
        = new HashMap<Integer, Matrix>();
  
    covarianceMatrices.put(0, COVARIANCE_0);
    covarianceMatrices.put(1, COVARIANCE_1);


    //Making arrays is too hard
    HDA filter = (HDA)getFilter();
    HashMap<Integer, HashMap<Integer, Matrix>> combinedScatters
        = filter.combineScatterMatrices(testProbabilities, covarianceMatrices);
    Matrix scatter_00 = combinedScatters.get(0).get(0);
    Matrix scatter_01 = combinedScatters.get(0).get(1);
    Matrix scatter_10 = combinedScatters.get(1).get(0);
    Matrix scatter_11 = combinedScatters.get(1).get(1);

    for (int i = 0; i < 2; ++i) {
      assertTrue(combinedScatters.containsKey(i));
    }
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        assertTrue(scatter_00.getArray()[i][j] >= ANS_00 - DELTA && 
                   scatter_00.getArray()[i][j] <= ANS_00 + DELTA);
        assertTrue(scatter_01.getArray()[i][j] >= ANS_01 - DELTA && 
                   scatter_01.getArray()[i][j] <= ANS_01 + DELTA);
        assertTrue(scatter_10.getArray()[i][j] >= ANS_10 - DELTA && 
                   scatter_10.getArray()[i][j] <= ANS_10 + DELTA);
        assertTrue(scatter_11.getArray()[i][j] >= ANS_11 - DELTA && 
                   scatter_11.getArray()[i][j] <= ANS_11 + DELTA);
      }
    }

  }
  
  public static Test suite() {
    return new TestSuite(HDATest.class);
  }

  public static void main(String[] args) {
    junit.textui.TestRunner.run(suite());
  }
}
