package weka.filters;

import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.*;
import weka.core.Option;
import weka.filters.SimpleBatchFilter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Vector;
import java.util.Collections;
import java.util.Enumeration;
import java.lang.Math;

import weka.core.Instances;
import weka.core.TestInstances;
import weka.filters.AbstractFilterTest;
import weka.filters.Filter;
import weka.filters.HDA;

import junit.framework.Test;
import junit.framework.TestSuite;
import org.junit.Assert;

public class HDATest
  extends AbstractFilterTest {

  private static final double DELTA = 1e-14;

  public HDATest(String name) {
    super(name);
  }

  protected void setUp() throws Exception {
    super.setUp();

    //m_Instances.deleteAttributeAt(6);  // Attribute Numeric type missing values
    //m_Instances.deleteAttributeAt(5);  // Attribute Date type missing values
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


  public static void assertArrayEquals(double[][] expected, 
                                       double[][] actual, 
                                       double delta) {
    if (actual == null && expected == null) {
      return;
    }

    if (actual.length != expected.length) {
      fail("The array dimensions are different.");
    }

    for (int i = 0; i < actual.length; ++i) {
      Assert.assertArrayEquals(expected[i], actual[i], delta);
    }
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

  public void testListOptions() {
    HDA hda = (HDA)getFilter();
    SimpleBatchFilter parent = (SimpleBatchFilter)getFilter();

    Vector<Option> hda_list = new Vector<Option>();
    Vector<Option> parent_list = new Vector<Option>();

    for (Enumeration<Option> parent_ops = parent.listOptions(); parent_ops.hasMoreElements();){
      parent_list.add(parent_ops.nextElement());
    }
    for (Enumeration<Option> hda_ops = hda.listOptions(); hda_ops.hasMoreElements();){
      hda_list.add(hda_ops.nextElement());
    }

    assertEquals(parent_list.size(), hda_list.size());

    for (int i = 0; i < hda_list.size(); ++i) {
      assertEquals(hda_list.get(i).description(), parent_list.get(i).description());
      assertEquals(hda_list.get(i).name(), parent_list.get(i).name());
      assertEquals(hda_list.get(i).numArguments(), parent_list.get(i).numArguments());
      assertEquals(hda_list.get(i).synopsis(), parent_list.get(i).synopsis());
    }
  }

  public void testSetOptions() {
    HDA filter = (HDA)getFilter();

    final int ANS = 3;
    final String[] op = {"-dim", 
                   "3", 
                   "-output-debug-info", 
                   "-do-not-check-capabilities"};
    
    try {
      filter.setOptions(op);
    } catch (Exception e) {
      fail("Exception thrown");
    }
    assertEquals(filter.getDimension(), ANS);
  }

  public void testGetOptions() {
    HDA hda = (HDA)getFilter();
    SimpleBatchFilter parent = (SimpleBatchFilter)getFilter();

    String[] hda_ops = hda.getOptions();
    String[] parent_ops = hda.getOptions();

    assertEquals(parent_ops.length, hda_ops.length);
    Assert.assertArrayEquals(parent_ops, hda_ops);
  }

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

  public void testCalculateRelativeProbability() {
    final double PROB_0 = 2;
    final double PROB_1 = 0.5;
    final double PROB_2 = 1;

    final double[][] ANS = {{0.5, 0.8, 2d/3},
                            {0.2, 0.5, 1d/3},
                            {1d/3, 2d/3, 0.5}};

    HashMap<Integer, Double> probabilities = new HashMap<Integer, Double>();
    probabilities.put(0, PROB_0);
    probabilities.put(1, PROB_1);
    probabilities.put(2, PROB_2);

    HDA filter = (HDA)getFilter();
    HashMap<Integer, HashMap<Integer, Double>> relativeProbabilities
            = filter.calculateRelativeProbability(probabilities);

    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        assertEquals(ANS[i][j], relativeProbabilities.get(i).get(j), DELTA);
      }
    }
  }
      
  public void testFindSampleMeans() {
      final Matrix MEAN_0 = new Matrix(0, 0);
      final Matrix MEAN_1 = new Matrix(3/2, 6/2);
      final Matrix MEAN_2 = new Matrix(12/3, 24/3);
      final Matrix MEAN_3 = new Matrix(30/4, 60/4);
      
      // 3 attributes
      ArrayList<Attribute> attinfo = new ArrayList<Attribute>();
      attinfo.add(new Attribute("example-ID"));
      attinfo.add(new Attribute("class-ID"));
      attinfo.add(new Attribute("example2-ID"));
      
      // Process the instances.
      HDA filter = (HDA)getFilter();
      HashMap<Integer, Instances> disjointDataset = new HashMap<Integer, Instances>();
      
      int id = 0;
      int id2 = 0;
      
      // Manually create a disjoint dataset. With 4 Instances each containing a different number of instance objects.
      // (1, 2, 3, and 4)
      for (int i = 0; i < 4; ++i) {
          Instances testInst = new Instances("Test instances", attinfo, 0);
          testInst.setClassIndex(0);
          
          for (int j = 0; j <= i; ++j) {
              Instance inst = new DenseInstance(3);
              inst.setValue(0, i);
              inst.setValue(1, id);
              inst.setValue(2, id2);
              ++id;
              id+=2;
              testInst.add(inst);
          }
          disjointDataset.put(i, testInst);
      }
      
      // Now that all the setup code is finished we can test findSampleMeans.
      HashMap<Integer, Matrix> sampleMeans = filter.findSampleMeans(disjointDataset);
      
      assertArrayEquals(MEAN_0.getArray(), sampleMeans.get(0).getArray(), DELTA);
      assertArrayEquals(MEAN_1.getArray(), sampleMeans.get(1).getArray(), DELTA);
      assertArrayEquals(MEAN_2.getArray(), sampleMeans.get(2).getArray(), DELTA);
      assertArrayEquals(MEAN_3.getArray(), sampleMeans.get(3).getArray(), DELTA);
  }

  public void testBetweenClassScatterMatrices() {
    final Matrix MEAN_0 = new Matrix(3, 1, 1);
    final Matrix MEAN_1 = new Matrix(3, 1, 1);

    final double[][] M_2 = {{1},
                            {3},
                            {5}};
    final Matrix MEAN_2 = new Matrix(M_2);

    final double[][] ZERO = {{0, 0, 0},
                             {0, 0, 0},
                             {0, 0, 0}};
    final double[][] ANS1 = {{0, 0, 0},
                             {0, 4, 8},
                             {0, 8, 16}};

    final double[][][][] EXPECTED = {{ZERO, ZERO, ANS1},
                                     {ZERO, ZERO, ANS1},
                                     {ANS1, ANS1, ZERO}};

    HashMap<Integer, Matrix> sampleMeans = new HashMap<Integer, Matrix>();
    sampleMeans.put(0, MEAN_0);
    sampleMeans.put(1, MEAN_1);
    sampleMeans.put(2, MEAN_2);

    HDA filter = (HDA)getFilter();
    HashMap<Integer, HashMap<Integer, Matrix>> betweenScatters
            = filter.betweenClassScatterMatrices(sampleMeans);

    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        assertArrayEquals(EXPECTED[i][j], 
              betweenScatters.get(i).get(j).getArray(), DELTA);
      }
    }
  }

  public void testCombineScatterMatrices() {
    final double PROB_00 = 0.753;
    final double PROB_01 = 3;
    final double PROB_10 = 1.75;
    final double PROB_11 = 2.33;
    final Matrix COVARIANCE_0 = new Matrix(3, 3, 0.8);
    final Matrix COVARIANCE_1 = new Matrix(3, 3, 2.73);

    final Matrix ANS_00 = new Matrix(3, 3, 1.2048);
    final Matrix ANS_01 = new Matrix(3, 3, 7.1775);
    final Matrix ANS_10 = new Matrix(3, 3, 7.1775);
    final Matrix ANS_11 = new Matrix(3, 3, 12.7218);

    final Matrix[][] EXPECTED = {{ANS_00, ANS_01},
                                 {ANS_10, ANS_11}};

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

    for (int i = 0; i < 2; ++i) {
      assertTrue(combinedScatters.containsKey(i));
    }
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        assertArrayEquals(EXPECTED[i][j].getArray(), combinedScatters.get(i).get(j).getArray(), DELTA);
      }
    }
  }

  public void testWithinClassScatterMatrix() {
    final double PROB_0 = 0.753;
    final double PROB_1 = 3;
    final Matrix COVARIANCE_0 = new Matrix(3, 3, 0.8);
    final Matrix COVARIANCE_1 = new Matrix(3, 3, 2.73);

    final double ANS = PROB_0 * COVARIANCE_0.get(0,0) + 
                       PROB_1 * COVARIANCE_1.get(0,0);
    final Matrix EXPECTED = new Matrix(3, 3, ANS);

    HashMap<Integer, Matrix> covariances = new HashMap<Integer, Matrix>();
    covariances.put(0, COVARIANCE_0);
    covariances.put(1, COVARIANCE_1);

    HashMap<Integer, Double> probabilities = new HashMap<Integer, Double>();
    probabilities.put(0, PROB_0);
    probabilities.put(1, PROB_1);

    HDA filter = (HDA)getFilter();
    Matrix withinClassScatter 
            = filter.withinClassScatterMatrix(covariances, probabilities);

    assertArrayEquals(EXPECTED.getArray(), withinClassScatter.getArray(), DELTA);

  }

  public void testMatrixToOneHalf() throws Exception {
    final double[][] TEST_1 = {{5, 4},
                               {4, 5}};
    final double[][] TEST_2 = {{1, 0},
                               {0, 1}};
    final Matrix M1 = new Matrix(TEST_1);
    final Matrix M2 = new Matrix(TEST_2);
    final Matrix M3 = new Matrix(TEST_2);
    final Matrix M4 = new Matrix(TEST_1);

    final double[][] ANS_1 = {{2, 1},
                              {1, 2}};
    final double[][] ANS_2 = {{1, 0},
                              {0, 1}};
    final double[][] ANS_3 = {{1, 0},
                              {0, 1}};
    final double[][] ANS_4 = {{2/3d, -1/3d},
                              {-1/3d, 2/3d}};

    final double[][][][] EXPECTED = {{ANS_1, ANS_2},
                                     {ANS_3, ANS_4}};

    HDA filter = (HDA)getFilter();
    Matrix half1 = filter.matrixToOneHalf(M1, true);
    Matrix half2 = filter.matrixToOneHalf(M2, true);
    Matrix half3 = filter.matrixToOneHalf(M3, false);
    Matrix half4 = filter.matrixToOneHalf(M4, false);
    Matrix[][] actual = {{half1, half2},
                         {half3, half4}};

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        assertArrayEquals(EXPECTED[i][j], actual[i][j].getArray(), DELTA);
      }
    }
  }

  public void testMatrixLog() throws Exception {
    final double[][] TEST = {{5, 4},
                             {4, 5}};
    final Matrix M = new Matrix(TEST);

    final double[][] EXPECTED = {{Math.log(9d)/2d, Math.log(9d)/2d},
                                 {Math.log(9d)/2d, Math.log(9d)/2d}};

    HDA filter = (HDA)getFilter();
    Matrix log = filter.matrixLog(M);

    assertArrayEquals(EXPECTED, log.getArray(), DELTA);
  }

  public void testSolutionIteration() throws Exception {
    final int I = 0;
    final int J = 1;

    final double REL_PROB = 1;
    final double PROB = 3;

    final Matrix withinClassScatter = Matrix.identity(2, 2);
    final Matrix B_SCATTER = Matrix.identity(2, 2);

    final double[][] C_SCAT = {{5, 4},
                               {4, 5}};
    final Matrix C_SCATTER = new Matrix(C_SCAT);

    final double[][] COVAR = {{5, 4}, 
                              {4, 5}};
    final Matrix COVARIANCE = new Matrix(COVAR);


    final double[][] ANS = {{5 - Math.log(9)*9/2, -4 - Math.log(9)*9/2},
                            {-4 - Math.log(9)*9/2, 5 - Math.log(9)*9/2}};
    final Matrix EXPECTED = new Matrix(ANS);
                            

    
    HashMap<Integer, HashMap<Integer, Matrix>> betweenClassScatter
        = new HashMap<Integer, HashMap<Integer, Matrix>>();
    betweenClassScatter.put(0, new HashMap<Integer, Matrix>());
    betweenClassScatter.put(1, new HashMap<Integer, Matrix>());
    betweenClassScatter.get(0).put(0, B_SCATTER);
    betweenClassScatter.get(0).put(1, B_SCATTER);
    betweenClassScatter.get(1).put(0, B_SCATTER);
    betweenClassScatter.get(1).put(1, B_SCATTER);
    
    HashMap<Integer, HashMap<Integer, Double>> relativeProbabilities
        = new HashMap<Integer, HashMap<Integer, Double>>();
    relativeProbabilities.put(0, new HashMap<Integer, Double>());
    relativeProbabilities.put(1, new HashMap<Integer, Double>());
    relativeProbabilities.get(0).put(0, REL_PROB);
    relativeProbabilities.get(0).put(1, REL_PROB);
    relativeProbabilities.get(1).put(0, REL_PROB);
    relativeProbabilities.get(1).put(1, REL_PROB);
   
    HashMap<Integer, HashMap<Integer, Matrix>> combinedScatters
        = new HashMap<Integer, HashMap<Integer, Matrix>>();
    combinedScatters.put(0, new HashMap<Integer, Matrix>());
    combinedScatters.put(1, new HashMap<Integer, Matrix>());
    combinedScatters.get(0).put(0, C_SCATTER);
    combinedScatters.get(0).put(1, C_SCATTER);
    combinedScatters.get(1).put(0, C_SCATTER);
    combinedScatters.get(1).put(1, C_SCATTER);
   
    HashMap<Integer, Matrix> covariances = new HashMap<Integer, Matrix>();
    covariances.put(0, COVARIANCE);
    covariances.put(1, COVARIANCE);

    HashMap<Integer, Double> probabilities = new HashMap<Integer, Double>();
    probabilities.put(0, PROB);
    probabilities.put(1, PROB);

    HDA filter = (HDA)getFilter();
    Matrix solution = filter.solutionIteration(I, J, withinClassScatter,
        betweenClassScatter, relativeProbabilities, combinedScatters,
        covariances, probabilities);

    assertArrayEquals(EXPECTED.getArray(), solution.getArray(), DELTA);
  }

  public void testReduceDimension() {
    HDA filter = (HDA)getFilter();
    final int NUM_CLASSES = 3;
    final int NUM_INSTANCES = 3;

    ArrayList<Attribute> attinfo = new ArrayList<Attribute>();
    attinfo.add(new Attribute("Attribute-1"));
    attinfo.add(new Attribute("Attribute-2"));
    attinfo.add(new Attribute("class-ID"));
    Instances testInst = new Instances("Test instances", attinfo, 0);
    testInst.setClassIndex(2);
    int id = 0;

    final double[][] reduce_values =
            {
              {31d,1.5d}
            };
    final Matrix EIGENVECTOR = new Matrix(reduce_values);
    final double[] expected_values = {63.5d, 37d, 63.5d};
    Instances EXPECTED = new Instances(testInst, 0);

    final double[][] instances_values =
            {
              {2d,1d,0d},
              {1d,4d,1d},
              {2d,1d,2d}
            };
    // Set up NUM_INSTANCES
    for (int j = 0; j < NUM_INSTANCES; ++j ) {
      Instance inst = new DenseInstance(3);
      inst.setValue(0, instances_values[j][0]);
      inst.setValue(1, instances_values[j][1]);
      inst.setValue(2, instances_values[j][2]);
      testInst.add(inst);

      Instance expectedInst = new DenseInstance(2);
      expectedInst.setValue(0, expected_values[j]);
      expectedInst.setValue(1, j);
      EXPECTED.add(expectedInst);
    }
    Instances ACTUAL = filter.reduceDimension(testInst, EIGENVECTOR);
    if (EXPECTED.numInstances() != ACTUAL.numInstances()) {
      fail("Reduction returned a incorrect amount of instances");
    }
    for (int i = 0; i < EXPECTED.numInstances(); ++i) {
      Assert.assertArrayEquals(EXPECTED.instance(i).toDoubleArray(),
                        ACTUAL.instance(i).toDoubleArray(),
                        DELTA);
    }
  }

  public void testSolution() throws Exception {

    final double REL_PROB = 1;
    final double PROB = 3;

    final Matrix withinClassScatter = Matrix.identity(2, 2);
    final Matrix B_SCATTER = Matrix.identity(2, 2);

    final double[][] C_SCAT = {{5, 4},
                               {4, 5}};
    final Matrix C_SCATTER = new Matrix(C_SCAT);

    final double[][] COVAR = {{5, 4},
                              {4, 5}};
    final double[][] SPECIAL_COVAR = {{2, 8},
                               {        1, 5}};
    final Matrix COVARIANCE = new Matrix(COVAR);
    final Matrix SPECIAL_COVARIANCE = new Matrix(SPECIAL_COVAR);


    // The matrix result was,
    // {
    //  { 11.9945444819427074, -91.8628690246541169},
    //  {-30.6344304013431241, -14.2462149280477064}
    // }
    //
    // This resulted in the eigen values
    // {-55.7729854800120535, 53.5213150339070722}
    //
    // With their associated matrix of eigen vectors in fractions and columns,
    // {
    //  {0.8047248011304596,  0.9112205391698998},
    //  {0.5936480391996102, -0.4119188378733325}
    // }
    // The 2nd is the largest, and we are using the default of one dimension.
    // These were found by using the expm package in R and solving the equation.
    final double[][] ANS = {{0.9112205391698998d, -0.4119188378733325d}};

    final Matrix EXPECTED = new Matrix(ANS);



    HashMap<Integer, HashMap<Integer, Matrix>> betweenClassScatter
        = new HashMap<Integer, HashMap<Integer, Matrix>>();
    betweenClassScatter.put(0, new HashMap<Integer, Matrix>());
    betweenClassScatter.put(1, new HashMap<Integer, Matrix>());
    betweenClassScatter.put(2, new HashMap<Integer, Matrix>());
    betweenClassScatter.get(0).put(0, B_SCATTER);
    betweenClassScatter.get(0).put(1, B_SCATTER);
    betweenClassScatter.get(0).put(2, B_SCATTER);
    betweenClassScatter.get(1).put(0, B_SCATTER);
    betweenClassScatter.get(1).put(1, B_SCATTER);
    betweenClassScatter.get(1).put(2, B_SCATTER);
    betweenClassScatter.get(2).put(0, B_SCATTER);
    betweenClassScatter.get(2).put(1, B_SCATTER);
    betweenClassScatter.get(2).put(2, B_SCATTER);

    HashMap<Integer, HashMap<Integer, Double>> relativeProbabilities
        = new HashMap<Integer, HashMap<Integer, Double>>();
    relativeProbabilities.put(0, new HashMap<Integer, Double>());
    relativeProbabilities.put(1, new HashMap<Integer, Double>());
    relativeProbabilities.put(2, new HashMap<Integer, Double>());
    relativeProbabilities.get(0).put(0, REL_PROB);
    relativeProbabilities.get(0).put(1, REL_PROB);
    relativeProbabilities.get(0).put(2, REL_PROB);
    relativeProbabilities.get(1).put(0, REL_PROB);
    relativeProbabilities.get(1).put(1, REL_PROB);
    relativeProbabilities.get(1).put(2, REL_PROB);
    relativeProbabilities.get(2).put(0, REL_PROB);
    relativeProbabilities.get(2).put(1, REL_PROB);
    relativeProbabilities.get(2).put(2, REL_PROB);

    HashMap<Integer, HashMap<Integer, Matrix>> combinedScatters
        = new HashMap<Integer, HashMap<Integer, Matrix>>();
    combinedScatters.put(0, new HashMap<Integer, Matrix>());
    combinedScatters.put(1, new HashMap<Integer, Matrix>());
    combinedScatters.put(2, new HashMap<Integer, Matrix>());
    combinedScatters.get(0).put(0, C_SCATTER);
    combinedScatters.get(0).put(1, C_SCATTER);
    combinedScatters.get(0).put(2, C_SCATTER);
    combinedScatters.get(1).put(0, C_SCATTER);
    combinedScatters.get(1).put(1, C_SCATTER);
    combinedScatters.get(1).put(2, C_SCATTER);
    combinedScatters.get(2).put(0, C_SCATTER);
    combinedScatters.get(2).put(1, C_SCATTER);
    combinedScatters.get(2).put(2, C_SCATTER);

    HashMap<Integer, Matrix> covariances = new HashMap<Integer, Matrix>();
    covariances.put(0, COVARIANCE);
    covariances.put(1, COVARIANCE);
    covariances.put(2, SPECIAL_COVARIANCE);

    HashMap<Integer, Double> probabilities = new HashMap<Integer, Double>();
    probabilities.put(0, PROB);
    probabilities.put(1, PROB);
    probabilities.put(2, PROB);

    HDA filter = (HDA)getFilter();
    Matrix solution = filter.solution(withinClassScatter,
        betweenClassScatter, relativeProbabilities, combinedScatters,
        covariances, probabilities);
    assertArrayEquals(EXPECTED.getArray(), solution.getArray(), DELTA);
  }

  public static Test suite() {
    return new TestSuite(HDATest.class);
  }

  public static void main(String[] args) {
    junit.textui.TestRunner.run(suite());
  }
}
